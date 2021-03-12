import json

class dbidReader():
    """
    Description:
        Class which can be used like a dictionary. It writes all the changes to the harddisk.
        You must dump the dbidReader, after you changed a variable in the dictionary through chained access (i.e dbid[alm_optimal][feature_extraction] = 0).
        Otherwise your changed variable wont be saved.
    Arguments:
        -path (string):     Path of the database
    """

    def __init__(self, path):
        self.path = f"{path}/dbid.json"

        #load in the dbid
        with open(self.path) as json_file:
            self.dbid = json.load(json_file)
        
    def __getitem__(self, key):
        try:
            return self.dbid[key]
        finally:
            self.dump()

    def __setitem__(self, key, item):
        #change the dict in the ram
        self.dbid[key] = item

        #save changes to json file
        with open(self.path, 'w') as fp:
            json.dump(self.dbid, fp,  indent=4)

        print("we got called")

    def dump(self):
        #save changes to json file
        with open(self.path, 'w') as fp:
            json.dump(self.dbid, fp,  indent=4)

dbid = dbidReader(".")

dbid["alm_optimal"]["feature_extraction"]["parameters"]["distance"] = "hallo"

print(dbid["alm_optimal"])