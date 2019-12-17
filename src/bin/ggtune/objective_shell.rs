use failure::ResultExt as _;
use ggtune::Space;
use itertools::Itertools as _;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::process::{Command, Stdio};
use std::str::FromStr;

pub struct RunCommandAsObjective {
    cli_template: Vec<String>,
    space: Space,
}

impl RunCommandAsObjective {
    pub fn new(cli_template: Vec<String>, space: Space) -> Self {
        Self {
            cli_template,
            space,
        }
    }
}

impl<A> ggtune::ObjectiveFunction<A> for RunCommandAsObjective
where
    A: Copy + FromStr + std::fmt::Display,
{
    fn run<'a>(&self, xs: &[ggtune::ParameterValue], rng: &'a mut ggtune::RNG) -> (A, A) {
        let template = self.cli_template.iter().map(String::as_ref).collect_vec();
        let seed = rng.uniform(0..=u32::max_value());
        run_command_as_objective(&self.space, template.as_slice(), xs, seed)
    }
}

fn run_command_as_objective<A>(
    space: &Space,
    template: &[&str],
    xs: &[ggtune::ParameterValue],
    seed: u32,
) -> (A, A)
where
    A: Copy + FromStr + std::fmt::Display,
{
    let mut args = collect_xs_as_hash(space, xs);
    args.insert("SEED".to_owned(), seed.to_string());
    let cmd_args = apply_template(template, &args)
        .expect("filling in objective command placeholders must succeed");
    let mut cmd_args = cmd_args.iter().map(AsRef::<OsStr>::as_ref);

    let output = Command::new(
        cmd_args
            .next()
            .expect("objective command needs a command name"),
    )
    .args(cmd_args)
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::inherit())
    .output()
    .expect("objective command failed to execute");

    assert!(
        output.status.success(),
        "objective command failed with status: {}",
        output.status
    );

    let output = String::from_utf8(output.stdout).expect("objective command output must be UTF-8");

    parse_command_output(&output)
        .with_context(|err| format!("while parsing command output: {}", err))
        .unwrap()
}

fn collect_xs_as_hash<A>(space: &Space, xs: &[A]) -> HashMap<String, String>
where
    A: Copy + std::fmt::Display,
{
    space
        .params()
        .iter()
        .zip_eq(xs)
        .map(|(param, &x)| (param.name().to_owned(), x.to_string()))
        .collect()
}

fn apply_template<A>(
    template: &[&str],
    args: &HashMap<String, A>,
) -> Result<Vec<String>, strfmt::FmtError>
where
    A: std::fmt::Display,
{
    template
        .iter()
        .map(|template| strfmt::strfmt(template, args))
        .collect()
}

fn parse_command_output<A>(output: &str) -> Result<(A, A), failure::Error>
where
    A: FromStr + std::fmt::Display,
{
    let last_line = output
        .lines()
        .filter(|line| !line.is_empty())
        .last()
        .ok_or_else(|| format_err!("at least one line is required"))?;

    let mut items = last_line.split_whitespace();

    let raw_value = items
        .next()
        .ok_or_else(|| format_err!("must contain value: {:?}", last_line))?;
    let value = raw_value
        .parse::<A>()
        .map_err(|_| format_err!("value `{}` must parse as a number", raw_value))?;

    let raw_cost = items.next().unwrap_or("0");
    let cost = raw_cost
        .parse::<A>()
        .map_err(|_| format_err!("cost `{}` must parse as a number", raw_cost))?;

    ensure!(
        items.next().is_none(),
        "last output line can only contain two items `<value> <cost>?`: {}",
        last_line
    );

    Ok((value, cost))
}

#[cfg(test)]
macro_rules! assert_err {
    ($expr:expr, $msg:expr $(, $($other:tt)* )?) => {
        assert_eq!(
            $expr.map_err(|err| err.to_string()),
            Err($msg.to_string()),
            $($($other)*)?
        )
    }
}

#[cfg(test)]
macro_rules! assert_ok {
    ($expr:expr, $value:expr $(, $($other:tt)*)?) => {
        assert_eq!(
            $expr.map_err(|err| err.to_string()),
            Ok($value),
            $($($other)*)?
        )
    }
}

#[test]
fn test_parse_command_output() {
    assert_err!(
        parse_command_output::<f64>(""),
        "at least one line is required",
    );

    // empty lines are ignored
    assert_ok!(parse_command_output("123\n\n"), (123.0, 0.0),);

    assert_err!(
        parse_command_output::<f64>("garbage\n "),
        r#"must contain value: " ""#,
    );

    assert_err!(
        parse_command_output::<f64>("foo bar"),
        "value `foo` must parse as a number",
    );

    assert_err!(
        parse_command_output::<f64>("123 xxx"),
        "cost `xxx` must parse as a number",
    );

    assert_err!(
        parse_command_output::<f64>("123 456 extra"),
        "last output line can only contain two items `<value> <cost>?`: 123 456 extra",
    );

    // success case
    assert_ok!(parse_command_output("foo bar\n 123 456 \n"), (123.0, 456.0),);
}
