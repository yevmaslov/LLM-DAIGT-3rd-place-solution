import re


symbols = [
    ';', '¨', '-', '`', '(', ')', '/', '$', '|', '=', '[', ']', '#', '+',
    '@', '\x9b', '´', '<', '>', '\\', '\xad', '¸', '~', '·', 'ï', '*', '\x92', 
    '\xa0', '¹', 'ó', '\x80', '\x85', '^', '¶', '£', '\x99', '{', '}', 'é', 
    'á', '²', 'ã', '\x82', 'â', '¡', 'º', '\x94', '\x83', '\x93', '©', '\x97', 
    'ë', '°', 'þ', 'ä', '\x88', '\x9f', '«', '¢', '®', 'å', '\x91', 
    '–', '’', '\u200b', '“', '”', 'ç', '\t', '€', '\x98', '\x9d', '‹', '—', 
    'ü', '\uefc0', '…', '�', '‘', '\x12', '\x02', '\x1d', 'ҽ', '页', '面', 
    '顶', '部', '\x13', 'к', '\x08', 'ö', '\x14', 'м', 'е', 'р', '\x05', 
    '\x19', '\x04', '\x1c', '❒', '\x0e', 'д', 'в', 'о', 'и', 'з', 'л', 
    'ь', 'ч', 'н', 'я', 'п', 'с', 'т', 'а', '\x1a', '\x06', 'й', '\x1e', 
    'у', 'г', 'х', '™', '\x17', '\x07', '︙', 'ы', 'ж', 'щ', 'ю', 'б', '\x18', 
    'э', '\x01', '′', '\x0f', '⇽', 'ц', 'ъ', '์', '₂', '❍', '″', 'í', 'ш', 'ф', 
    'ё', 'џ', '\x7f', '\x1f', '‟', '\x1b', '\x15', 'à', 'न', 'ह', 'ी', 'ं', '„', 
    '\x11', '³', '：', '有', '时', '他', '会', '用', '惩', '罚', '手', '段', '来', 
    '引', '导', '儿', '童', '。', '⌁', '\x03', 'ʼ'
]   
  


def normalize_characters(text):
    quotes = ["''", '“', '”', "¨", '‟', '″']
    for quote in quotes:
        text = text.replace(quote, '"')
        
    single_quotes = ['ʼ', '`', '´', '′', '`', '´', '‘', '’',]
    for quote in single_quotes:
        text = text.replace(quote, "'")
        
    subtract_sign = ['–', '—', '-', '-']
    for sign in subtract_sign:
        text = text.replace(sign, " ")
        
    punctuation_signs = ['¸']
    for sign in punctuation_signs:
        text = text.replace(sign, ',')
        
    punctuation_signs = ['︙', '：']
    for sign in punctuation_signs:
        text = text.replace(sign, ':')
        
    punctuation_signs = ['…', '. . .']
    for sign in punctuation_signs:
        text = text.replace(sign, '...')
        
    return text
    
    
def clean_text(text):
    text = text.strip()
    text = re.sub(r"([.,:;!?'])", r' \1 ', text)
    text = re.sub(r'(["])', r' \1 ', text)
    text = re.sub(r'\r\n', r'\n\n', text)
    text = re.sub(r'\n+', r'', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    replace_with = ' '
    
    text = re.sub(r'[()\[\]{}]', replace_with, text)
    text = re.sub(r'[<>]', replace_with, text)
    text = text.replace('-', replace_with)

    text = normalize_characters(text)
    
    for symbol in symbols:
        text = text.replace(symbol, replace_with)
        
    return text.strip()

    
    
def clean_text2(text):
    text = text.strip()
    text = re.sub(r"([.,:;!?'])", r' \1 ', text)
    text = re.sub(r'(["])', r' \1 ', text)
    text = re.sub(r'\r\n', r'\n\n', text)
    text = re.sub(r'\n+', r'\n', text)
    
    replace_with = ' '
    
    text = re.sub(r'[()\[\]{}]', replace_with, text)
    text = re.sub(r'[<>]', replace_with, text)
    text = text.replace('-', replace_with)
    
    text = normalize_characters(text)
    
    for symbol in symbols:
        text = text.replace(symbol, replace_with)
        
    return text.strip()


def make_text(dataframe, config):    
    dataframe['full_text'] = dataframe['text']
    return dataframe