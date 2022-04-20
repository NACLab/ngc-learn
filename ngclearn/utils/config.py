import os
import sys
import copy

class Config:
    """
    Simple configuration object to house named arguments for experiments

    File format is:
    # Comments start with pound symbol
    arg_name = arg_value
    arg_name = arg_value # side comment that will be stripped off

    @author: Alex Ororbia
    """

    def __init__(self, fname):
        self.fname = fname
        self.variables = {}

        # read in text file to convert to query-able configuration object
        fd = open(fname, 'r')
        count = 0
        while True:
            count += 1
            line = fd.readline() # get next line from file
            if not line: # if line is empty end of file (EOF) has been reached
                break
            line = line.replace(" ", "").replace("\n", "")
            if len(line) > 0:
                cmt_split = line.split("#")
                argmt = cmt_split[0]
                if (len(argmt) > 0) and ("=" in argmt):
                    tok = argmt.split("=")
                    var_name = tok[0]
                    var_val = tok[1]
                    #print("{0}  ->  {1}".format(var_name,var_val))
                    self.variables[var_name] = var_val
                # else, ignore comment-only lines
            # else, ignore empty lines
        fd.close()

    def getArg(self, arg_name):
        """
            Retrieve argument from current configuration
        """
        return self.variables.get(arg_name)

    def hasArg(self, arg_name):
        """
            Check if argument exists (or if it is known by this config object)
        """
        arg = self.variables.get(arg_name)
        flag = False
        if arg is not None:
            flag = True
        return flag

    def setArg(self, arg_name, arg_value):
        """
            Set argument directly
        """
        self.variables[arg_name] = arg_value
