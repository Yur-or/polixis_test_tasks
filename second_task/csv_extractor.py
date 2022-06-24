from typing import Optional, Tuple
import itertools
import time
import json
import csv
import os


def split_by_list_delimeters(text: str, delimiters: list[str]) -> list[str]:

    first_delimiter = delimiters[0]
    for delimiter in delimiters[1:]:
        text = text.replace(delimiter, first_delimiter)
    
    return text.split(first_delimiter)


def create_pairs_with_ls(first_word: str, ls_second_words: list[str]) -> list[Tuple[str, str]]:

    return [(first_word, second_word) for second_word in ls_second_words]


def pairs_extractor_v1(table_links: list[Tuple[str, str]], delimiters: list[str]) -> Tuple[list[Tuple[str, str]], list[Tuple[str, str]]]:
    
    first_lang_first_lang_pairs = []
    first_lang_second_lang_pairs = []

    for row in table_links:
        ls_words_first_lang = split_by_list_delimeters(row[0], delimiters)
        ls_words_second_lang = split_by_list_delimeters(row[1], delimiters)

        for i, word_first_lang in enumerate(ls_words_first_lang):
            first_lang_first_lang_pairs.extend(create_pairs_with_ls(word_first_lang, ls_words_first_lang[i+1:]))
            first_lang_second_lang_pairs.extend(create_pairs_with_ls(word_first_lang, ls_words_second_lang))
    
    return first_lang_first_lang_pairs, first_lang_second_lang_pairs


def pairs_extractor_v2(table_links: list[Tuple[str, str]], delimiters: list[str]) -> Tuple[list[Tuple[str, str]], list[Tuple[str, str]]]:
    
    first_lang_first_lang_pairs = []
    first_lang_second_lang_pairs = []

    for row in table_links:
        ls_words_first_lang = split_by_list_delimeters(row[0], delimiters)
        ls_words_second_lang = split_by_list_delimeters(row[1], delimiters)
        first_lang_first_lang_pairs.extend(itertools.combinations(ls_words_first_lang, 2))
        first_lang_second_lang_pairs.extend(itertools.product(ls_words_first_lang, ls_words_second_lang))

    return first_lang_first_lang_pairs, first_lang_second_lang_pairs


def read_csv_as_list_rows(path_to_csv: os.PathLike, delimiter: str = ',') -> Optional[list[Tuple[str, str]]]:

    try:
        with open(path_to_csv) as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader)
            ls_rows = [tuple(row)[1:] for row in reader]
        
        return ls_rows
    except Exception as e:
        print(f'Failed to convert file was thrown exception {e.message, e.args}')
        return None


def main():
    path_to_csv = 'second_task/data/name_equivalence.csv'
    delimiters = ['،', ';']
    
    csv_as_ls_rows = read_csv_as_list_rows(path_to_csv)

    start_time = time.time()
    for i in range(1000000):
        first_lang_first_lang_pairs_v1, first_lang_second_lang_pairs_v1 = pairs_extractor_v1(csv_as_ls_rows, delimiters)
    print(f'Avg time to process by version 1: {(time.time() - start_time) / 1000000}')

    start_time = time.time()
    for i in range(1000000):
        first_lang_first_lang_pairs_v2, first_lang_second_lang_pairs_v2 = pairs_extractor_v2(csv_as_ls_rows, delimiters)
    print(f'Avg time to process by version 2: {(time.time() - start_time) / 1000000}')

    print(f'Versions 1 and 2 given equal results: {first_lang_first_lang_pairs_v1 == first_lang_first_lang_pairs_v2} {first_lang_second_lang_pairs_v1 == first_lang_second_lang_pairs_v2}')


    # with open('outputs/first_lang_first_lang_pairs.json', 'w', encoding='utf-8') as f:
    #     json.dump(first_lang_first_lang_pairs_v2, f, ensure_ascii=False, indent=4)
    
    # with open('outputs/first_lang_second_lang_pairs.json', 'w', encoding='utf-8') as f:
    #     json.dump(first_lang_second_lang_pairs_v2, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
