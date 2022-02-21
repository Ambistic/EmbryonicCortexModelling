import re

def nonestr(string):
    if string is None:
        return ""
    return string

class StringModel:
    def __init__(self, model, default=dict(), global_default="", sep="_.", escape="$", forbidden="", replace="%"):
        """
        The escape parameter shall never be used in the values
        This class is not made for proper handling of escape mechanism
        """
        assert len(escape) == 1
        self.model = model
        self.default = default
        self.global_default = global_default
        self.sep = sep
        self.escape = escape
        self.forbidden = forbidden
        self.replace = replace
        
    def _split(self, string):
        pat = f"(?:^|([^{self.escape}]))[{self.sep}]"
        res = re.split(pat, string)
        size = int((len(res) + 1) / 2)
        res += ['']
        return [res[i * 2] + nonestr(res[i * 2 + 1]) for i in range(size)]
    
    def _parse_substring(self, substring):
        pat ='([\w\d]+)(?:{([\w\d]+)})?'
        return re.findall(pat, substring)[0]
    
    def _get_slots(self):
        substrings = self._split(self.model)
        return [self._parse_substring(substring) for substring in substrings]
    
    def _process_dict_strings(self, my_dict):
        for char in self.forbidden:
            for k, v in my_dict.items():
                my_dict[k] = str(v).replace(char, self.replace)
        return my_dict
    
    def fill(self, **params):
        params = self._process_dict_strings(params)
        values = {slotname: self.global_default for _, slotname in self._get_slots()}
        values.update(self.default)
        values.update(params)
        return self.model.format(**values)
    
    def _extract_slot(self, string, prefix):
        pat = f"(?:^|[{self.sep}])({prefix}[^{self.sep}]*)(?:$|[{self.sep}])"
        return re.findall(pat, string)[0]
    
    def extract(self, string):
        slots = self._get_slots()
        substrings = self._split(string)
        assert len(slots) == len(substrings), f"Are you sure {string} matches the model"
        dict_res = dict()
        for (prefix, slotname), substring in zip(slots, substrings):
            if slotname == "":
                continue
            if re.match(f"{prefix}.*", substring):
                dict_res[slotname] = re.findall(f"{prefix}(.*)", substring)[0]

        return dict_res