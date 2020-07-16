import click
import pandas as pd
from tqdm.auto import tqdm
from googletrans import Translator


@click.command()
@click.option('--input_path', type=click.STRING, help='Path to input file')
@click.option('--output_path', type=click.STRING, help='Path to input file')
@click.option('--set_', type=click.Choice(['train', 'test']), help="set")
@click.option('--pivot', type=click.Choice(['en', 'fr', 'pt', 'ar']), help="pivot leng")
def translate(input_path, output_path, set_, pivot):
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

    sentences_es = list(df.Pregunta.values)
    print(f'Amount of sentences {len(sentences_es)}')

    translator = Translator()

    translations = []
    for sent in tqdm(sentences_es):
        translation = translator.translate(sent, src="es", dest=pivot).text
        translations.append(translation)
    print(f'Amount sentences en: {len(translations)}')

    translations_es_back = []
    for sent in tqdm(translations):
        translation = translator.translate(sent, src=pivot, dest="es").text
        translations_es_back.append(translation)
    print(f'Amount sentences en: {len(translations_es_back)}')

    df[f'Pregunta_{pivot}'] = translations_es_back
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    translate()
