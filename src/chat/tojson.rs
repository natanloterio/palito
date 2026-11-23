//! The `tojson` filter, with the same bytes as llama.cpp's templates: spaced
//! separators, keys in their original order and no HTML escaping.

use minijinja::{
    Error, ErrorKind, Value,
    filters,
    value::Kwargs,
};
use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};
use std::io::{self, Write};

/// `, ` between items and `: ` after keys, as Python's `json.dumps`.
struct Spaced;

impl Formatter for Spaced {
    fn begin_array_value<W: ?Sized + Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Calls with options (`indent`) keep minijinja's implementation.
pub(super) fn tojson(value: &Value, indent: Option<Value>, args: Kwargs) -> Result<Value, Error> {
    if indent.is_some() || args.args().next().is_some() {
        return filters::tojson(value, indent, args);
    }
    let mut bytes = Vec::new();
    value
        .serialize(&mut Serializer::with_formatter(&mut bytes, Spaced))
        .map_err(|error| {
            Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(error)
        })?;
    let text = String::from_utf8(bytes).map_err(|error| {
        Error::new(ErrorKind::InvalidOperation, "JSON is not UTF-8").with_source(error)
    })?;
    Ok(Value::from_safe_string(text))
}
