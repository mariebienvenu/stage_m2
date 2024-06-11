
import os, json


class AbstractIO:

    def __init__(self, directory:str, verbose=0):
        self.directory = directory
        self.verbose = verbose


    def finalize_init(self, maker):
        self.config = None
        self.config_loaded = False
        self.is_processed = False
        try:
            self.load_config()
            self.complete_config(maker)
            if self.verbose>0: print("Config file found.")
        except OSError:
            if self.verbose>0: print("Did not find config file ; reverting to default config.")
            self.make_default_config(maker)
        assert self.config_loaded, f"Error when finalizing {type(self).__name__} object."
        self.save_config()


    def make_default_config(self, maker):
        self.config = maker()
        self.config_loaded = True

    
    def complete_config(self, maker):
        default:dict = maker()
        for key, value in default.items():
            if key not in self.config:
                self.config[key] = value
            if type(value) is dict:
                for key2, value2 in value.items():
                    if key2 not in self.config[key]:
                        self.config[key][key2] = value2
            if type(value) is list and len(value)==1 and type(value[0]) is dict:
                for index in range(len(self.config[key])):
                    for key2, value2 in value[0].items():
                        if key2 not in self.config[key][index]:
                            self.config[key][index][key2] = value2

    
    def save_config(self):
        assert self.config_loaded, "Cannot save config when no config is loaded."
        with open(self.config_filename, "w") as outfile:
            json.dump(self.config, outfile) #, indent=4)


    def load_config(self, force=False):
        if self.config_loaded and not force:
            return
        if not os.path.exists(self.config_filename):
            raise OSError("No config file.")
        with open(self.config_filename, 'r') as openfile:
            self.config = json.load(openfile)
        self.config_loaded = True


    @property
    def config_filename(self):
        raise NotImplementedError


    def process(self, force=False):
        raise NotImplementedError
    

    def __repr__(self):
        return f'{type(self).__name__}({self.config_filename}, {self.config})'