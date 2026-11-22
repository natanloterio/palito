use super::*;

fn sampler(temperature: f32, top_p: f32, top_k: u32, seed: u64) -> Sampler {
    with(Sampling {
        temperature,
        top_p,
        top_k,
        seed: Some(seed),
        ..Sampling::GREEDY
    })
}

fn with(settings: Sampling) -> Sampler {
    Sampler::new(settings).unwrap()
}

fn counts(sampler: &mut Sampler, logits: &[f32], draws: usize) -> Vec<usize> {
    let mut counts = vec![0; logits.len()];
    for _ in 0..draws {
        counts[sampler.sample(logits).unwrap() as usize] += 1;
    }
    counts
}

#[test]
fn greedy_settings_pick_the_argmax() {
    let logits = [0.1, 3.0, 2.9, -1.0];
    for mut s in [
        sampler(0.0, 1.0, 0, 1),
        sampler(1.0, 1.0, 1, 1),
        sampler(1.0, 1e-6, 0, 1),
    ] {
        for _ in 0..50 {
            assert_eq!(s.sample(&logits).unwrap(), 1);
        }
    }
}

#[test]
fn same_seed_same_tokens_and_temperature_follows_softmax() {
    let logits: Vec<f32> = (0..64).map(|i| (i % 7) as f32 * 0.3).collect();
    let draw = |seed| {
        let mut s = sampler(0.8, 1.0, 0, seed);
        (0..40).map(|_| s.sample(&logits).unwrap()).collect::<Vec<_>>()
    };
    assert_eq!(draw(7), draw(7));
    // ln 3 apart at temperature 1: the larger is chosen ~3x as often.
    let two = [0.0, 3f32.ln()];
    let c = counts(&mut sampler(1.0, 1.0, 0, 3), &two, 40_000);
    let ratio = c[1] as f64 / c[0] as f64;
    assert!((ratio - 3.0).abs() < 0.15, "ratio {ratio}");
}

#[test]
fn top_k_and_top_p_drop_the_tail() {
    let logits = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let c = counts(&mut sampler(1.0, 1.0, 2, 5), &logits, 5_000);
    assert!(c[0] > 0 && c[1] > 0 && c[2..].iter().all(|&n| n == 0));
    // Probabilities ~0.636, 0.234, 0.086...: top_p 0.8 keeps two tokens.
    let c = counts(&mut sampler(1.0, 0.8, 0, 9), &logits, 5_000);
    assert!(c[0] > 0 && c[1] > 0 && c[2..].iter().all(|&n| n == 0), "{c:?}");
}

#[test]
fn top_p_is_taken_before_temperature() {
    // At temperature 1, top_p 0.8 keeps two tokens (see above). A high
    // temperature flattens the draw but must not widen the nucleus (llama.cpp
    // applies temperature last).
    let logits = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let c = counts(&mut sampler(5.0, 0.8, 0, 9), &logits, 20_000);
    assert!(c[0] > 0 && c[1] > 0 && c[2..].iter().all(|&n| n == 0), "{c:?}");
    // Two kept tokens one logit apart at temperature 5: ratio e^(1/5).
    let ratio = c[0] as f64 / c[1] as f64;
    assert!((ratio - 0.2f64.exp()).abs() < 0.1, "ratio {ratio}");
}

#[test]
fn min_p_keeps_tokens_near_the_most_likely() {
    // Relative probabilities 1, e^-1 (0.37), e^-2 (0.14), e^-3 (0.05).
    let logits = [3.0, 2.0, 1.0, 0.0];
    let min_p = |min_p, top_k, top_p| Sampling {
        temperature: 1.0,
        min_p,
        top_k,
        top_p,
        seed: Some(4),
        ..Sampling::GREEDY
    };
    let c = counts(&mut with(min_p(0.2, 0, 1.0)), &logits, 5_000);
    assert!(c[0] > 0 && c[1] > 0 && c[2] == 0 && c[3] == 0, "{c:?}");
    let c = counts(&mut with(min_p(0.1, 0, 1.0)), &logits, 5_000);
    assert!(c[2] > 0 && c[3] == 0, "{c:?}");
    // After top-k and top-p, and relative to the untempered distribution.
    let c = counts(&mut with(min_p(0.1, 2, 1.0)), &logits, 5_000);
    assert!(c[1] > 0 && c[2] == 0, "{c:?}");
    let c = counts(&mut with(Sampling { temperature: 8.0, ..min_p(0.2, 0, 0.99) }), &logits, 5_000);
    assert!(c[0] > 0 && c[1] > 0 && c[2] == 0 && c[3] == 0, "{c:?}");
    // min_p 1 keeps only the most likely token.
    assert!(counts(&mut with(min_p(1.0, 0, 1.0)), &logits, 500)[1..].iter().all(|&n| n == 0));
}

#[test]
fn nucleus_falls_back_to_an_exact_pass_for_flat_distributions() {
    // One dominant token and a huge flat tail below the prefilter.
    // exp(-14) = 8.3e-7 per tail token: 500k of them hold ~29% of the mass.
    let mut logits = vec![-14.0f32; 500_000];
    logits[0] = 0.0;
    let mut s = sampler(1.0, 0.99, 0, 11);
    let c = counts(&mut s, &logits, 300);
    let tail: usize = c[1..].iter().sum();
    assert!((40..140).contains(&tail), "tail drawn {tail} of 300 times");
}

#[test]
fn penalties_apply_to_the_window_of_prompt_and_generated_tokens() {
    let penalized = |repeat_penalty, frequency_penalty, presence_penalty, repeat_last_n| Sampling {
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        repeat_last_n,
        ..Sampling::GREEDY
    };
    let logits = [2.0, 1.9, 0.5, -1.0];
    // Greedy picks 0 until it is penalized: 2 / 1.5 = 1.33 < 1.9.
    let mut s = with(penalized(1.5, 0.0, 0.0, 64));
    assert_eq!(s.sample(&logits).unwrap(), 0);
    s.begin(&[3, 0]);
    assert_eq!(s.sample(&logits).unwrap(), 1);
    // Both seen: 1.33 vs 1.27.
    s.accept(1);
    assert_eq!(s.sample(&logits).unwrap(), 0);
    // A window of one only sees the last token.
    let mut s = with(penalized(1.5, 0.0, 0.0, 1));
    s.begin(&[0, 1]);
    assert_eq!(s.sample(&logits).unwrap(), 0);
    // Frequency counts occurrences, presence does not: 2 - 3 * 0.1 = 1.7 < 1.9;
    // 2 - 0.05 = 1.95 > 1.9.
    let mut s = with(penalized(1.0, 0.1, 0.0, 64));
    s.begin(&[0, 0, 0]);
    assert_eq!(s.sample(&logits).unwrap(), 1);
    let mut s = with(penalized(1.0, 0.0, 0.05, 64));
    s.begin(&[0, 0, 0]);
    assert_eq!(s.sample(&logits).unwrap(), 0);
    // Negative logits move further down: -1 * 3 = -3 < -2.
    let mut s = with(penalized(3.0, 0.0, 0.0, 64));
    s.begin(&[3]);
    let negative = [-2.0, -2.5, -2.9, -1.0];
    assert_eq!(s.sample(&negative).unwrap(), 0);
    // repeat_last_n 0 turns the penalties off.
    let mut s = with(penalized(10.0, 1.0, 1.0, 0));
    s.begin(&[0]);
    assert_eq!(s.sample(&logits).unwrap(), 0);
}

#[test]
fn logit_bias_shifts_and_bans_tokens() {
    let biased = |logit_bias: Vec<(u32, f32)>| Sampling { logit_bias, ..Sampling::GREEDY };
    let logits = [2.0, 1.9, 0.5];
    assert_eq!(with(biased(vec![(2, 1.6)])).sample(&logits).unwrap(), 2);
    assert_eq!(with(biased(vec![(0, f32::NEG_INFINITY)])).sample(&logits).unwrap(), 1);
    let mut banned = with(Sampling {
        temperature: 1.0,
        seed: Some(1),
        ..biased(vec![(1, f32::NEG_INFINITY)])
    });
    assert_eq!(counts(&mut banned, &logits, 2_000)[1], 0);
    let all = (0..3).map(|t| (t, f32::NEG_INFINITY)).collect();
    assert!(with(biased(all)).sample(&logits).is_err());
}

#[test]
fn only_plain_greedy_skips_host_logits() {
    assert!(!Sampling::GREEDY.needs_host_logits());
    assert!(!Sampling { min_p: 0.5, repeat_last_n: 0, repeat_penalty: 1.3, ..Sampling::GREEDY }
        .needs_host_logits());
    for settings in [
        Sampling { temperature: 0.7, ..Sampling::GREEDY },
        Sampling { repeat_penalty: 1.1, ..Sampling::GREEDY },
        Sampling { presence_penalty: 0.5, ..Sampling::GREEDY },
        Sampling { frequency_penalty: -0.5, ..Sampling::GREEDY },
        Sampling { logit_bias: vec![(0, 0.0)], ..Sampling::GREEDY },
    ] {
        assert!(settings.needs_host_logits(), "{settings:?}");
    }
}

#[test]
fn rejects_bad_settings_and_nonfinite_logits() {
    for (t, p) in [(-0.1, 1.0), (f32::NAN, 1.0), (1.0, 0.0), (1.0, 1.5)] {
        assert!(Sampler::new(Sampling { temperature: t, top_p: p, ..Sampling::GREEDY }).is_err());
    }
    for bad in [
        Sampling { min_p: -0.1, ..Sampling::GREEDY },
        Sampling { min_p: 1.5, ..Sampling::GREEDY },
        Sampling { min_p: f32::NAN, ..Sampling::GREEDY },
        Sampling { repeat_penalty: 0.0, ..Sampling::GREEDY },
        Sampling { repeat_penalty: 1e-39, ..Sampling::GREEDY },
        Sampling { repeat_penalty: f32::INFINITY, ..Sampling::GREEDY },
        Sampling { presence_penalty: f32::NAN, ..Sampling::GREEDY },
        Sampling { frequency_penalty: f32::NEG_INFINITY, ..Sampling::GREEDY },
        Sampling { logit_bias: vec![(0, f32::NAN)], ..Sampling::GREEDY },
        Sampling { logit_bias: vec![(0, f32::INFINITY)], ..Sampling::GREEDY },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
    let biased = Sampling { logit_bias: vec![(3, -1.0)], ..Sampling::GREEDY };
    assert!(biased.check_vocabulary(4).is_ok() && biased.check_vocabulary(3).is_err());
    assert!(sampler(1.0, 1.0, 0, 1).sample(&[0.0, f32::INFINITY]).is_err());
}
