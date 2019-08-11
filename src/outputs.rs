extern crate prettytable;

use crate::maybe_owned::MaybeOwned;
use crate::{Individual, Space, SurrogateModel};
use failure::ResultExt as _;
use std::boxed::Box;
use std::time::Duration;

/// Report progress and save results during optimization process.
pub trait OutputEventHandler<A> {
    /// Called when a new generation is started.
    fn event_new_generation(&mut self, _gen: usize, _relscale: &[f64]) {}

    /// Called when evaluations of a generation has completed.
    fn event_evaluations_completed(&mut self, _individuals: &[Individual<A>], _duration: Duration) {
    }

    /// Called when a new model has been trained
    fn event_model_trained(
        &mut self,
        _gen: usize,
        _model: &dyn SurrogateModel<A>,
        _duration: Duration,
    ) {
    }

    /// Called when new samples have been acquired.
    fn event_acquisition_completed(&mut self, _duration: Duration) {}
}

impl<A> OutputEventHandler<A> for Output<'_, A> {
    fn event_new_generation(&mut self, gen: usize, relscale: &[f64]) {
        for logger in &mut self.subloggers {
            logger.as_mut().event_new_generation(gen, relscale);
        }
    }

    fn event_evaluations_completed(&mut self, individuals: &[Individual<A>], duration: Duration) {
        for logger in &mut self.subloggers {
            logger
                .as_mut()
                .event_evaluations_completed(individuals, duration);
        }
    }

    fn event_model_trained(
        &mut self,
        gen: usize,
        model: &dyn SurrogateModel<A>,
        duration: Duration,
    ) {
        for logger in &mut self.subloggers {
            logger.as_mut().event_model_trained(gen, model, duration);
        }
    }

    fn event_acquisition_completed(&mut self, duration: Duration) {
        for logger in &mut self.subloggers {
            logger.as_mut().event_acquisition_completed(duration);
        }
    }
}

pub struct HumanReadableIndividualsOutput<'life> {
    writer: Box<dyn std::io::Write + 'life>,
    param_names: Vec<String>,
}

impl<'life> HumanReadableIndividualsOutput<'life> {
    pub fn new(writer: impl std::io::Write + 'life, space: &Space) -> Self {
        let writer = Box::new(writer);
        let param_names = space.params().iter().map(|p| p.name().to_owned()).collect();
        Self {
            writer,
            param_names,
        }
    }
}

impl<A> OutputEventHandler<A> for HumanReadableIndividualsOutput<'_>
where
    A: Copy + std::fmt::Display,
{
    fn event_evaluations_completed(&mut self, individuals: &[Individual<A>], duration: Duration) {
        use prettytable::format::*;
        use prettytable::{Cell, Row, Table};

        let mut titles = Row::empty();
        titles.add_cell(Cell::new("gen"));
        titles.add_cell(Cell::new("observation"));
        titles.add_cell(Cell::new("prediction"));
        titles.add_cell(Cell::new("ei"));
        titles.add_cell(Cell::new("cost"));
        titles.extend(self.param_names.iter());

        let mut table = Table::new();
        table.set_titles(titles);

        table.set_format(
            FormatBuilder::new()
                .column_separator(' ')
                .separator(LinePosition::Top, LineSeparator::new('-', '-', '-', '-'))
                .separator(LinePosition::Title, LineSeparator::new('-', ' ', ' ', ' '))
                .separator(LinePosition::Bottom, LineSeparator::new('-', '-', '-', '-'))
                .build(),
        );

        for ind in individuals {
            let row = table.add_empty_row();
            let prec2 = |maybex: Option<_>| {
                maybex
                    .map(|x| format!("{:.2}", x))
                    .unwrap_or_else(|| "--.--".to_owned())
            };

            let genstr = ind
                .gen()
                .map(|gen| gen.to_string())
                .unwrap_or_else(|| "--".to_owned());

            let cell_right = |text: String| Cell::new_align(&text, Alignment::RIGHT);

            row.add_cell(cell_right(genstr));
            row.add_cell(cell_right(prec2(ind.observation())));
            row.add_cell(cell_right(prec2(ind.prediction())));
            row.add_cell(cell_right(prec2(ind.expected_improvement())));
            row.add_cell(cell_right(prec2(ind.cost())));
            for x in ind.sample() {
                row.add_cell(cell_right(format!("{:.02}", x)));
            }
        }

        write!(
            self.writer,
            "evaluation completed in {}s:\n{}",
            duration.as_millis() as f64 / 1e3,
            table,
        )
        .with_context(|err| format!("could not write evaluations: {}", err))
        .unwrap();
    }
}

#[test]
fn test_human_readable_individuals_output() {
    let mut ind = Individual::new(vec![0.123, 12.2]);
    ind.set_gen(3).unwrap();
    ind.set_cost(0.759).unwrap();
    ind.set_expected_improvement(0.178).unwrap();
    ind.set_observation(15.399).unwrap();
    ind.set_prediction(16.981).unwrap();

    let mut space = Space::new();
    space.add_real_parameter("x", 0.0, 100.0);
    space.add_real_parameter("y", 0.0, 100.0);

    let mut buffer = Vec::new();
    {
        let mut output = HumanReadableIndividualsOutput::new(&mut buffer, &space);
        output.event_evaluations_completed(&[ind], Duration::from_millis(170));
    }
    let actual = String::from_utf8(buffer).expect("wrote correct utf8");
    let expected = r#"evaluation completed in 0.17s:
-----------------------------------------------
gen observation prediction ei   cost x    y
--- ----------- ---------- ---- ---- ---- -----
  3       15.40      16.98 0.18 0.76 0.12 12.20
-----------------------------------------------
"#;
    assert_eq!(
        actual, expected,
        "\nactual:\n{}\nexpected:\n{}\nend",
        actual, expected
    );
}

#[derive(Default)]
pub struct DurationCounter {
    evaluation_durations: Vec<Duration>,
    training_durations: Vec<Duration>,
    acquisition_durations: Vec<Duration>,
}

impl<A> OutputEventHandler<A> for DurationCounter {
    fn event_evaluations_completed(&mut self, _individuals: &[Individual<A>], duration: Duration) {
        self.evaluation_durations.push(duration);
    }

    fn event_model_trained(
        &mut self,
        _gen: usize,
        _model: &dyn SurrogateModel<A>,
        duration: Duration,
    ) {
        self.training_durations.push(duration);
    }

    fn event_acquisition_completed(&mut self, duration: Duration) {
        self.acquisition_durations.push(duration)
    }
}

pub struct Output<'life, A> {
    subloggers: Vec<MaybeOwned<'life, dyn OutputEventHandler<A>>>,
}

impl<A> Default for Output<'_, A> {
    fn default() -> Self {
        Self {
            subloggers: Vec::new(),
        }
    }
}

impl<'life, A> Output<'life, A> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add(&mut self, logger: impl OutputEventHandler<A> + 'life) {
        self.subloggers.push(MaybeOwned::Owned(Box::new(logger)));
    }

    pub fn add_borrowed(&mut self, logger: &'life mut dyn OutputEventHandler<A>) {
        self.subloggers.push(MaybeOwned::Borrowed(logger));
    }

    pub fn add_duration_counter(&mut self, counter: &'life mut DurationCounter) {
        self.add_borrowed(counter);
    }
}
