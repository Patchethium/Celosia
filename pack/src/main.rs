use clap::Parser;
mod en;
use en::pack_en;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
  /// The language to use
  #[arg(short, long, default_value = "en")]
  lang: String,
  /// The path to the assets
  #[arg(short, long, default_value = "assets")]
  assets: String,
  /// The output path
  #[arg(short, long, default_value = "assets/en.pack.zst")]
  output: String,
}

fn main() {
  let args = Args::parse();
  let lang = args.lang;
  let assets = args.assets;
  let output = args.output;
  match lang.as_str() {
    "en" => {
      pack_en(&assets, &output).unwrap();
    }
    _ => {
      eprintln!("Unsupported language: {}", lang);
      return;
    }
  }
}
