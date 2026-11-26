//! `palito pull`, `palito list`, `palito rm`, and `--model <name>` resolution.
//!
//! Human-readable output goes to stderr; stdout carries only the model path
//! (or `--json`), so `palito chat --model "$(palito pull qwen3-8b)"` works.

use super::{
    ModelRef, exit,
    registry::{self, Entry, Registry},
    store::{Installed, Store},
};
use crate::cli_ui::{self, fail, human_size};
use clap::Args;
use palito::{Error, Result};
use serde_json::json;
use std::{
    io::{BufRead, IsTerminal, Write},
    path::PathBuf,
};

const MODEL_EXTENSIONS: [&str; 2] = [".palito", ".gguf"];

#[derive(Args)]
pub(crate) struct PullArgs {
    /// Model name (see 'palito list --available'), user/repo or user/repo:file.
    #[arg(value_name = "MODEL")]
    model: String,
    /// Download again even if the model is up to date.
    #[arg(short, long)]
    force: bool,
    /// Hash the downloaded file again and download it again if it does not match.
    #[arg(long, conflicts_with = "force")]
    verify: bool,
    /// Branch, tag or commit, for user/repo [default: the registry pin, or main].
    #[arg(long, value_name = "REV")]
    revision: Option<String>,
    /// Print only errors (and the path on stdout).
    #[arg(short, long, conflicts_with = "json")]
    quiet: bool,
    /// Print one JSON object on stdout.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
pub(crate) struct ListArgs {
    /// List every model in the registry, not only downloaded ones.
    #[arg(short, long)]
    available: bool,
    /// Fetch the registry again instead of using the cached copy.
    #[arg(long, requires = "available")]
    refresh: bool,
    /// Print a JSON array on stdout.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
pub(crate) struct RmArgs {
    /// Downloaded models to delete (names from 'palito list').
    #[arg(value_name = "MODEL", required = true)]
    models: Vec<String>,
    /// Delete without asking.
    #[arg(short, long)]
    yes: bool,
    /// Show what would be deleted.
    #[arg(long)]
    dry_run: bool,
}

/// What `pull` did.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Downloaded,
    UpToDate,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Downloaded => "downloaded",
            Self::UpToDate => "up_to_date",
        }
    }
}

struct PullOptions<'a> {
    force: bool,
    verify: bool,
    revision: Option<&'a str>,
    quiet: bool,
}

fn not_found(model: &ModelRef, registry: &Registry) -> Error {
    let mut text = format!("unknown model '{model}'");
    if let ModelRef::Name(name) = model
        && let Some(close) = registry.suggest(name)
    {
        text.push_str(&format!("\n  hint: did you mean '{close}'?"));
    }
    text.push_str(
        "\n  hint: see 'palito list --available', or pull any Hugging Face repo: palito pull user/repo",
    );
    fail(exit::NOT_FOUND, text)
}

/// Registry entry for `model`, if any (a `user/repo` can match a pinned entry).
fn registry_entry(model: &ModelRef, registry: &Registry) -> Option<Entry> {
    match model {
        ModelRef::Name(name) => registry.find(name).cloned(),
        ModelRef::Repo { repo, file } => registry
            .models
            .iter()
            .find(|e| e.repo == *repo && file.as_ref().is_none_or(|f| *f == e.file))
            .cloned(),
    }
}

fn already_have(store: &Store, entry: &Entry) -> Option<Installed> {
    store
        .find(&ModelRef::Repo { repo: entry.repo.clone(), file: Some(entry.file.clone()) })
        .into_iter()
        .next()
}

#[cfg(not(feature = "hub"))]
fn pull_model(_model: &ModelRef, _options: &PullOptions) -> Result<(Installed, Status)> {
    Err(Error::Invalid(
        "this palito was built without download support\n  \
         hint: rebuild with --features hub, or download the .palito file yourself and pass its path"
            .into(),
    ))
}

#[cfg(feature = "hub")]
fn pull_model(model: &ModelRef, options: &PullOptions) -> Result<(Installed, Status)> {
    use super::{download::Hub, progress::Progress, store::Manifest};
    let store = Store::from_env();
    let registry = registry::refreshed(false);
    let entry = match (registry_entry(model, &registry), model) {
        (Some(entry), _) if options.revision.is_none() => Some(entry),
        (_, ModelRef::Name(_)) => return Err(not_found(model, &registry)),
        _ => None,
    };
    if let Some(entry) = entry.as_ref().filter(|e| !e.supported()) {
        return Err(Error::Invalid(format!(
            "{} needs palito {} or newer (this is {})\n  hint: update palito: https://github.com/natanloterio/palito/releases",
            entry.name,
            entry.min_palito.as_deref().unwrap_or_default(),
            env!("CARGO_PKG_VERSION")
        )));
    }
    let hub = Hub::new()?;
    // Repos outside the registry: pick the file and read its metadata.
    let (repo, revision, file) = match (&entry, model) {
        (Some(e), _) => (e.repo.clone(), e.revision.clone(), e.file.clone()),
        (None, ModelRef::Repo { repo, file }) => {
            let revision = options.revision.unwrap_or("main").to_owned();
            if !super::valid_segment(&revision) {
                return Err(Error::Invalid(format!("'{revision}' is not a valid revision")));
            }
            let file = match file {
                Some(file) => file.clone(),
                None => pick_file(repo, &hub.list_files(repo, &revision)?)?,
            };
            (repo.clone(), revision, file)
        }
        (None, ModelRef::Name(_)) => unreachable!("names without an entry returned above"),
    };
    let existing = store
        .find(&ModelRef::Repo { repo: repo.clone(), file: Some(file.clone()) })
        .into_iter()
        .next();
    if let Some(installed) = existing.as_ref().filter(|_| !options.force) {
        let wanted = entry.as_ref().map(|e| e.sha256.as_str());
        let current = installed.manifest.sha256.as_deref();
        let same_version = wanted.is_none_or(|w| current == Some(w));
        let intact = !options.verify || {
            if !options.quiet {
                eprintln!("Checking {}...", installed.path.display());
            }
            current.is_some_and(|c| super::download::sha256_file(&installed.path).is_ok_and(|h| h == c))
        };
        if same_version && intact {
            return Ok((installed.clone(), Status::UpToDate));
        }
        if !intact && !options.quiet {
            cli_ui::warn(&format!("{} does not match its checksum; downloading it again", installed.manifest.file));
        }
    }
    let mut remote = hub.head(&repo, &revision, &file)?;
    let expected_sha = entry.as_ref().map(|e| e.sha256.clone()).or(remote.sha256.clone());
    if remote.size.is_none() {
        remote.size = entry.as_ref().map(|e| e.size);
    }
    let commit = remote.commit.clone().unwrap_or_else(|| revision.clone());
    let name = entry.as_ref().map_or_else(|| stem(&file).to_owned(), |e| e.name.clone());
    if !options.quiet {
        eprintln!("Pulling {name} from {repo} (rev {})", short(&commit));
    }
    if expected_sha.is_none() && !options.quiet {
        cli_ui::warn("no checksum published for this file; it will not be verified");
    }
    let dir = store.repo_dir(&repo);
    let mut progress = Progress::new(&file, options.quiet);
    let digest = hub.download(&remote, &dir, &file, expected_sha.as_deref(), &mut progress)?;
    if expected_sha.is_some() && !options.quiet {
        eprintln!("  sha256 {} ok", short_hash(&digest));
    }
    for extra in entry.iter().flat_map(|e| &e.extra_files) {
        let fetched = hub.head(&repo, &revision, extra).and_then(|remote| {
            hub.download(&remote, &dir, extra, None, &mut Progress::new(extra, true))
        });
        if let Err(error) = fetched
            && !options.quiet
        {
            cli_ui::warn(&format!("could not download {extra}: {}", cli_ui::message(&error)));
        }
    }
    let mut manifest = match &entry {
        Some(entry) => Manifest::from_entry(entry),
        None => Manifest {
            name: name.clone(),
            aliases: Vec::new(),
            repo: repo.clone(),
            revision: commit.clone(),
            file: file.clone(),
            size: 0,
            sha256: None,
            license: String::new(),
            notice: None,
            pulled_at: 0,
        },
    };
    manifest.revision = commit;
    manifest.sha256 = Some(digest);
    manifest.size = std::fs::metadata(dir.join(&file)).map_or(0, |m| m.len());
    manifest.pulled_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    store.write_manifest(&manifest)?;
    let path = dir.join(&file);
    Ok((Installed { manifest, path }, Status::Downloaded))
}

#[cfg(feature = "hub")]
fn pick_file(repo: &str, files: &[String]) -> Result<String> {
    let models: Vec<&String> = files.iter().filter(|f| f.ends_with(".palito") && !f.contains('/')).collect();
    match models.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(fail(
            exit::NOT_FOUND,
            format!(
                "{repo} has no .palito file\n  hint: name a file explicitly: palito pull {repo}:<file>.gguf"
            ),
        )),
        many => Err(Error::Invalid(format!(
            "{repo} has {} .palito files; choose one:{}",
            many.len(),
            many.iter().map(|f| format!("\n  palito pull {repo}:{f}")).collect::<String>()
        ))),
    }
}

fn stem(file: &str) -> &str {
    MODEL_EXTENSIONS.iter().find_map(|ext| file.strip_suffix(ext)).unwrap_or(file)
}

fn short(commit: &str) -> &str {
    if commit.len() == 40 { &commit[..7] } else { commit }
}

fn short_hash(hash: &str) -> String {
    if hash.len() > 13 { format!("{}…{}", &hash[..8], &hash[hash.len() - 5..]) } else { hash.to_owned() }
}

/// License line printed after a pull and on stderr when a notice exists.
fn print_license(installed: &Installed) {
    let manifest = &installed.manifest;
    if let Some(notice) = &manifest.notice {
        eprintln!("{notice}");
        let license = installed.path.with_file_name("LICENSE");
        if license.is_file() {
            eprintln!("  License: {}", license.display());
        }
    }
}

pub(crate) fn pull(args: PullArgs) -> Result<()> {
    let model = ModelRef::parse(&args.model)?;
    let options = PullOptions {
        force: args.force,
        verify: args.verify,
        revision: args.revision.as_deref(),
        quiet: args.quiet || args.json,
    };
    let (installed, status) = pull_model(&model, &options)?;
    let manifest = &installed.manifest;
    if !options.quiet {
        match status {
            Status::UpToDate => eprintln!(
                "{} is up to date ({}, rev {})",
                manifest.name,
                human_size(manifest.size),
                short(&manifest.revision)
            ),
            Status::Downloaded => {
                let license = if !manifest.license.is_empty() && manifest.notice.is_none() {
                    format!(" ({})", manifest.license)
                } else {
                    String::new()
                };
                eprintln!("Pulled {}{license}. Run it: palito chat --model {}", manifest.name, manifest.name);
                print_license(&installed);
            }
        }
    }
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "schema_version": 1,
                "status": status.as_str(),
                "name": manifest.name,
                "repo": manifest.repo,
                "revision": manifest.revision,
                "file": manifest.file,
                "path": installed.path,
                "size": manifest.size,
                "sha256": manifest.sha256,
                "license": manifest.license,
            }))?
        );
    } else {
        println!("{}", installed.path.display());
    }
    Ok(())
}

fn table(rows: &[Vec<String>]) {
    let widths: Vec<usize> = (0..rows[0].len())
        .map(|col| rows.iter().map(|r| r[col].chars().count()).max().unwrap_or(0))
        .collect();
    let mut out = std::io::stdout().lock();
    for row in rows {
        let line: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                if i + 1 == row.len() { cell.clone() } else { format!("{cell:<w$}", w = widths[i]) }
            })
            .collect();
        let _ = writeln!(out, "{}", line.join("  ").trim_end());
    }
}

pub(crate) fn list(args: ListArgs) -> Result<()> {
    let store = Store::from_env();
    let installed = store.installed();
    if args.available {
        let registry = if args.refresh { registry::refreshed(true) } else { registry::refreshed(false) };
        if args.json {
            let models: Vec<_> = registry
                .models
                .iter()
                .map(|e| json!({
                    "name": e.name, "aliases": e.aliases, "description": e.description,
                    "repo": e.repo, "file": e.file, "size": e.size, "license": e.license,
                    "downloaded": already_have(&store, e).is_some(), "supported": e.supported(),
                }))
                .collect();
            println!("{}", serde_json::to_string(&models)?);
            return Ok(());
        }
        let mut rows = vec![vec!["NAME".into(), "SIZE".into(), "LICENSE".into(), "DESCRIPTION".into()]];
        for entry in &registry.models {
            let mut description = entry.description.clone();
            if already_have(&store, entry).is_some() {
                description.push_str("  [downloaded]");
            }
            if !entry.supported() {
                description.push_str(&format!("  [needs palito {}]", entry.min_palito.as_deref().unwrap_or("")));
            }
            rows.push(vec![entry.name.clone(), human_size(entry.size), entry.license.clone(), description]);
        }
        table(&rows);
        return Ok(());
    }
    if args.json {
        let models: Vec<_> = installed
            .iter()
            .map(|i| json!({
                "name": i.manifest.name, "repo": i.manifest.repo, "revision": i.manifest.revision,
                "file": i.manifest.file, "path": i.path, "size": i.manifest.size,
                "sha256": i.manifest.sha256, "license": i.manifest.license,
            }))
            .collect();
        println!("{}", serde_json::to_string(&models)?);
        return Ok(());
    }
    if installed.is_empty() {
        eprintln!(
            "No models in {} yet. Try: palito pull qwen3-8b  (all models: palito list --available)",
            store.root().display()
        );
        return Ok(());
    }
    let registry = Registry::local();
    let mut rows = vec![vec!["NAME".into(), "SIZE".into(), "LICENSE".into(), "SOURCE".into()]];
    for item in &installed {
        let manifest = &item.manifest;
        let newer = registry.find(&manifest.name).is_some_and(|e| {
            e.repo == manifest.repo && manifest.sha256.as_deref().is_some_and(|s| s != e.sha256)
        });
        let source = if newer { format!("{}  (update available)", manifest.repo) } else { manifest.repo.clone() };
        rows.push(vec![manifest.name.clone(), human_size(manifest.size), manifest.license.clone(), source]);
    }
    table(&rows);
    Ok(())
}

fn confirm(question: &str, default_yes: bool) -> bool {
    eprint!("{question} {} ", if default_yes { "[Y/n]" } else { "[y/N]" });
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    if std::io::stdin().lock().read_line(&mut answer).is_err() {
        return false;
    }
    match answer.trim().to_ascii_lowercase().as_str() {
        "" => default_yes,
        "y" | "yes" | "s" | "sim" => true,
        _ => false,
    }
}

fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

pub(crate) fn rm(args: RmArgs) -> Result<()> {
    let store = Store::from_env();
    let mut targets = Vec::new();
    for text in &args.models {
        let model = ModelRef::parse(text)?;
        let found = store.find(&model);
        if found.is_empty() {
            return Err(fail(
                exit::NOT_FOUND,
                format!("'{model}' is not downloaded\n  hint: see 'palito list'"),
            ));
        }
        targets.extend(found);
    }
    for target in targets {
        let manifest = &target.manifest;
        let dir = store.repo_dir(&manifest.repo);
        let what = format!("{} ({}) from {}", manifest.name, human_size(manifest.size), dir.display());
        if args.dry_run {
            eprintln!("Would remove {what}");
            continue;
        }
        if !args.yes {
            if !interactive() {
                return Err(Error::Invalid(
                    "refusing to delete without confirmation\n  hint: pass --yes".into(),
                ));
            }
            if !confirm(&format!("Remove {what}?"), false) {
                eprintln!("Kept {}", manifest.name);
                continue;
            }
        }
        let freed = store.remove(&target)?;
        eprintln!("Removed {} ({} freed)", manifest.name, human_size(freed));
    }
    Ok(())
}

/// Turns `--model <value>` into a file path. An existing path always wins;
/// otherwise a downloaded model name or `user/repo` is looked up, and a
/// known but missing model is downloaded (asking first on a terminal,
/// without asking with `--pull`).
pub(crate) fn resolve_model(model: &mut PathBuf, pull: bool) -> Result<()> {
    if model.exists() {
        return Ok(());
    }
    let Some(text) = model.to_str().map(str::to_owned) else { return Ok(()) };
    let looks_like_file = MODEL_EXTENSIONS.iter().any(|ext| text.ends_with(ext)) && !text.contains(':');
    if looks_like_file {
        return Err(missing_file_with_hint(&text));
    }
    let Ok(reference) = ModelRef::parse(&text) else { return Ok(()) };
    let store = Store::from_env();
    match store.find(&reference).as_slice() {
        [one] => {
            *model = one.path.clone();
            if let Some(notice) = &one.manifest.notice
                && std::io::stderr().is_terminal()
            {
                eprintln!("{notice}");
            }
            return Ok(());
        }
        [] => {}
        many => {
            return Err(Error::Invalid(format!(
                "'{reference}' matches {} downloads; choose one:{}",
                many.len(),
                many.iter()
                    .map(|i| format!("\n  --model {}:{}", i.manifest.repo, i.manifest.file))
                    .collect::<String>()
            )));
        }
    }
    // Names the cached registry does not know may be newer than the cache.
    let mut registry = Registry::local();
    let mut entry = registry_entry(&reference, &registry);
    if entry.is_none() && matches!(reference, ModelRef::Name(_)) {
        registry = registry::refreshed(false);
        entry = registry_entry(&reference, &registry);
    }
    if entry.is_none() && matches!(reference, ModelRef::Name(_)) {
        return Err(not_found_or_file(&reference, &registry));
    }
    let size = entry.as_ref().map_or(String::new(), |e| {
        format!(" ({} from {}, {})", human_size(e.size), e.repo, e.license)
    });
    if !pull {
        if !interactive() {
            return Err(fail(
                exit::NOT_FOUND,
                format!(
                    "model '{reference}' is not downloaded\n  \
                     hint: run 'palito pull {reference}', or add --pull to download it automatically"
                ),
            ));
        }
        eprintln!("{reference} is not downloaded yet{size}.");
        if !confirm("Download it now?", true) {
            return Err(fail(exit::NOT_FOUND, format!("model '{reference}' is not downloaded")));
        }
    }
    let options = PullOptions { force: false, verify: false, revision: None, quiet: false };
    let (installed, _) = pull_model(&reference, &options)?;
    print_license(&installed);
    *model = installed.path;
    Ok(())
}

/// A bare word that is neither a file nor a known model.
fn not_found_or_file(reference: &ModelRef, registry: &Registry) -> Error {
    let error = not_found(reference, registry);
    Error::Invalid(format!(
        "{}\n  hint: or pass a .gguf or .palito file with --model (or set PALITO_MODEL)",
        cli_ui::message(&error)
    ))
}

/// `--model qwen3-8b.palito` that does not exist: point at the name.
fn missing_file_with_hint(text: &str) -> Error {
    let base = text.rsplit('/').next().unwrap_or(text);
    let name = stem(base);
    let mut message = format!(
        "model file not found: {text}\n  hint: pass a .gguf or .palito file with --model (or set PALITO_MODEL)"
    );
    if Registry::local().find(name).is_some() {
        message.push_str(&format!("\n  hint: to download it: palito pull {name}, then use --model {name}"));
    }
    Error::Invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_stems_and_short_forms() {
        assert_eq!(stem("qwen3-8b.palito"), "qwen3-8b");
        assert_eq!(stem("m.gguf"), "m");
        assert_eq!(short("0123456789012345678901234567890123456789"), "0123456");
        assert_eq!(short("main"), "main");
        assert_eq!(short_hash(&"ab".repeat(32)), "abababab…babab");
    }

    #[test]
    fn missing_palito_file_points_at_the_registry_name() {
        let error = missing_file_with_hint("models/qwen3-8b.palito");
        let text = cli_ui::message(&error);
        assert!(text.starts_with("model file not found: models/qwen3-8b.palito"));
        assert!(text.contains("palito pull qwen3-8b"));
        let other = cli_ui::message(&missing_file_with_hint("x.gguf"));
        assert!(!other.contains("palito pull"));
    }

    #[test]
    fn existing_paths_are_left_alone() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let before = path.clone();
        resolve_model(&mut path, false).unwrap();
        assert_eq!(path, before);
    }

    #[cfg(feature = "hub")]
    #[test]
    fn picks_the_only_palito_file() {
        let files = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(pick_file("u/r", &files(&["README.md", "m.palito"])).unwrap(), "m.palito");
        assert!(pick_file("u/r", &files(&["README.md"])).is_err());
        let many = cli_ui::message(&pick_file("u/r", &files(&["a.palito", "b.palito"])).unwrap_err());
        assert!(many.contains("palito pull u/r:a.palito"));
    }
}
