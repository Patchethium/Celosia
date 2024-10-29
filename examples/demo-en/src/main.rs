#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use celosia::en::Phonemizer;
use eframe::egui;

fn main() -> eframe::Result {
  let options = eframe::NativeOptions {
    viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
    ..Default::default()
  };
  eframe::run_native(
    "My egui App",
    options,
    Box::new(|_| Ok(Box::<DemoApp>::default())),
  )
}

struct DemoApp {
  sentence: String,
  phonemes: Vec<Vec<&'static str>>,
  phonemizer: Phonemizer,
}

impl Default for DemoApp {
  fn default() -> Self {
    Self {
      sentence: "".to_owned(),
      phonemes: vec![],
      phonemizer: Phonemizer::default(),
    }
  }
}

impl eframe::App for DemoApp {
  fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    egui::CentralPanel::default().show(ctx, |ui| {
      ui.heading("Phonemizer Demo");

      ui.horizontal(|ui| {
        ui.label("Type your sentence here: ");
      });
      if ui
        .add(
          egui::widgets::text_edit::TextEdit::singleline(&mut self.sentence)
            .desired_width(ui.available_width()),
        )
        .changed()
      {
        self.phonemes = self.phonemizer.phonemize(&self.sentence);
      }
      ui.heading("Phonemes:");
      if !self.phonemes.is_empty() {
        for phoneme in &self.phonemes {
          ui.horizontal(|ui| {
            for p in phoneme {
              ui.label(*p);
            }
          });
        }
      } else {
        ui.label("Type a sentence to see its phonemes.");
      }
    });
  }
}
