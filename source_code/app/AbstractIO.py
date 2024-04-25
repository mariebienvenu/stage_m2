
import os, json


class AbstractIO:

    def __init__(self, directory, verbose=0):
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
        assert self.config_loaded, "Error when finalizing AbstractIO object."
        self.save_config()


    def make_default_config(self, maker):
        self.config = maker()
        self.config_loaded = True

    
    def complete_config(self, maker):
        default:dict = maker()
        for key, value in default.items():
            if key not in self.config:
                self.config[key] = value

    
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
        return f'AbstractIO({self.config_filename}), {self.config})'