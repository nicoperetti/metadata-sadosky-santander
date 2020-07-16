import click
import pandas as pd
# Due textacy problems
try:
    from textacy.preprocess import preprocess_text
except Exception:
    from textacy.preprocess import preprocess_text


def preprocess_f(text, fix_unicode=True, lowercase=True,
                 no_urls=True, no_emails=True,
                 no_phone_numbers=True,
                 no_numbers=True, no_currency_symbols=True,
                 no_punct=True, no_accents=True):
    """Preprocess text."""
    clean_text = preprocess_text(text, fix_unicode=fix_unicode,
                                 lowercase=lowercase,
                                 no_urls=no_urls, no_emails=no_emails,
                                 no_phone_numbers=no_phone_numbers,
                                 no_numbers=no_numbers,
                                 no_currency_symbols=no_currency_symbols,
                                 no_punct=no_punct,
                                 no_accents=no_accents)
    return clean_text


@click.command()
@click.option('--input_path', type=click.STRING, help='Path to input file')
@click.option('--output_path', type=click.STRING, help='Path to input file')
@click.option('--set_', type=click.Choice(['train', 'test']), help="set")
def preprocess(input_path, output_path, set_):
    """pre-process script

    :param input_path: path to input file
    :type input_path: str
    :param output_path: path to output file
    :type output_path: str
    :param set_: kind of data
    :type set_: str
    """
    if set_ == "train":
        df = pd.read_csv(input_path, sep='|')
    else:
        df = pd.read_csv(input_path)

    df["clean_txt"] = df["Pregunta"].apply(lambda x: preprocess_f(x))

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    preprocess()
