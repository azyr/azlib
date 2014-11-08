"""
Collection of tools for natural language processing (NLP)
"""
import re
import Levenshtein


def normalize_security_name(s):
    """Normalize a security name for NLP.

    Input is security name (str), output is normalized security name (str).
    """
    s = s.lower()
    s = re.sub(r'\bholding\b', 'hldg', s)
    s = re.sub(r'\bholdings\b', 'hldg', s)
    s = re.sub(r'\bcompany\b', 'co', s)
    s = re.sub(r'\bincorporated\b', 'inc', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&quot;', '"', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&copy;', '©', s)
    s = re.sub(r'&reg;', '®', s)
    s = re.sub(r'&pound;', '£', s)
    s = re.sub(r'&euro;', '€', s)
    s = s.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    s = s.replace('{', '').replace('}', '')
    return s


def deabbreviate(strlist, templates):
    if type(strlist) is str:
        strlist = [strlist]
    for i in range(len(strlist)):
        s = strlist[i]
        if re.match('[^.]+[.]$', s):
            base = s[:-1]
            re_base = re.compile("^{}.*".format(base))
            matches = [template for template in templates if re_base.match(template)]
            if len(matches) == 1:
                strlist[i] = matches[0]
    return strlist


def get_setratio(s1, s2):
    """Calculate similarity ratio of strings splitted in set of words.

    Returns a ratio between 0 and 1.

    Arguments:
    s1 - first string
    s2 - second string
    """
    s1 = normalize_security_name(s1)
    s2 = normalize_security_name(s2)
    s1 = list(set(s1.split()))
    s2 = list(set(s2.split()))
    s1 = deabbreviate(s1, s2)
    s2 = deabbreviate(s2, s1)
    return Levenshtein.setratio(s1, s2)
