import os
import json
from datetime import datetime
import difflib
import codecs
import functools
import itertools


def save_file(content, filename, encoding='utf-8', newline='\n'):
    with open(filename, 'w', encoding=encoding, newline=newline) as outfile:
        outfile.write(content)


def save_json(json_dict, json_file):
    with open(json_file, 'w') as outfile:
        json.dump(json_dict, outfile)


def process_file(folder, filename, func, **kwargs):
    file_path = os.path.join(folder, filename)
    return func(file_path, **kwargs)


def file_exists_decorator(func):
    """
    Decorator
    :param func:
    :return:
    """
    @functools.wraps(func)
    def wrapper(self, file):
        if not os.path.exists(file):
            return ''
        else:
            return func(self, file)

    return wrapper


def list_dir(folder, **kwargs):
    """
    Utility to retrieve files and/or folders from a given source folder, recursively or not,
    possibly filtered by certain specs
    :param folder: the folder to scan
    :param kwargs: options to filter-out elements of certain type or to keep only elements with given extension
    :return: a list of full paths of all the elements in folder
    """

    dir_only = kwargs.get("dir_only", False)
    files_only = kwargs.get("files_only", False)
    extension_filter = kwargs.get("extension_filter", "")
    assert not (dir_only and files_only and extension_filter), "Arguments dir_only, files_only and extension_filter are mutually exclusive"
    apply_recursively = kwargs.get("apply_recursively", False)

    # scan folder recursively adding all the elements in the folder
    dir_contents = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        dir_contents.append(os.path.join(folder, name))
        if apply_recursively and os.path.isdir(path):
            dir_contents.extend(list_dir(folder=os.path.join(folder, name), **kwargs))

    # keep only the elements compliant to the filters specified
    if dir_only:
        dir_contents = [path for path in dir_contents if os.path.isdir(path)]
    elif files_only:
        dir_contents = [path for path in dir_contents if os.path.isfile(path)]
    elif extension_filter:
        dir_contents = [path for path in dir_contents if path.endswith(extension_filter)]

    return dir_contents


def apply_recursively(folder, func, file_extension='.txt', **kwargs):
    """

    :param folder:
    :param func:
    :param file_extension:
    :param kwargs:
    :return: a list of outputs
    """

    outputs = []
    for name in os.listdir(folder):
        if name.endswith(file_extension):
            outputs.append(func(os.path.join(folder, name), **kwargs))
        elif os.path.isdir(name):
            outputs.extend(apply_recursively(os.path.join(folder, name), func, file_extension, **kwargs))

    return outputs


def decorate_str_with_date(str_value):
    now = datetime.now()
    now = now.strftime("%d_%m_%Y_%H_%M_%S")
    return f"{str_value}_{now}"


def create_dir_if_missing(folder):
    """
    :param folder: a string with the path of the folder
    """

    if not os.path.exists(folder):
        os.mkdir(folder)


def join_and_create_if_missing(current_path, folder_name):
    folder = os.path.join(current_path, folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    return folder


def file_len(fname):
    """
    Get the number of lines in a file
    :param fname: file path
    :return: number of lines in a file. 0 if the file not exists, 1 if empty.
    """
    if os.path.exists(fname):
        with codecs.open(fname, "r", encoding="utf-8", errors="replace") as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    else:
        return 0


def is_loose_alias(alias1, alias2):
    alias1_chunks = set(get_alias_chunks(alias1, min_len=5, min_words=1))
    alias2_chunks = set(get_alias_chunks(alias2, min_len=5, min_words=1))

    shared_chunks = alias1_chunks & alias2_chunks
    if len(shared_chunks) > 0:
        return True

    return False


def is_excluded_word(w):
    excluded_words = [
        "Company", "LLC", "Inc.", "Inc", "Inc,", "INC", "INC.", "INC,",
        "US", "LLP", "Co", "De", "The", "Las", "llc", "llp",
        "Group", "Limited", "Management", "Fund", "technologies", "Sdn", "Bhd", "Srl", "analytics", "Solutions",
        "Enterprise", "ltd", "avogados", "studio", "&", "plc", "associati", "spa", "legal", "financial"
    ]
    excluded_words = [word.lower() for word in excluded_words]

    return w.lower() in excluded_words


def all_stop_words(ws):
    for w in ws.split():
        if not is_excluded_word(w):
            return False
    return True


def get_alias_chunks(alias, min_len=5, min_all_upper_word_len=3, min_words=1):
    alias_words = alias.replace(",", '').strip().split()
    combinations = [alias, alias.lower(), alias.upper()]
    _max_alias_words = 15
    alias_words = alias_words[:_max_alias_words]
    for i in range(len(alias_words)-1, min_words, -1):
        i_sized_combinations = [' '.join(combination).strip() for combination in list(itertools.combinations(alias_words, i))]
        i_sized_combinations = [i_sized_comb for i_sized_comb in i_sized_combinations if len(i_sized_comb) >= min_len or i_sized_comb[0].isupper() and min_len>3]
        combinations.extend(i_sized_combinations)

    combinations = combinations + [w for w in alias_words if w.isupper() and len(w) >= min_all_upper_word_len and min_words == 1]
    cleaned_alias = [w.strip() for w in alias.replace(",", '').split() if len(w.strip()) >= min_len]
    combinations += cleaned_alias

    combinations = [combination.lower() for combination in combinations if not is_excluded_word(combination)]
    combinations = [combination.lower() for combination in combinations if not all_stop_words(combination)]

    return combinations
