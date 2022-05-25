import os
import sys
import copy

class Config:
    """
    Simple configuration object to house named arguments for experiments (to be
    built from a .cfg file on disk).

    | File format is:
    | # Comments start with pound symbol
    | arg_name = arg_value
    | arg_name = arg_value # side comment that will be stripped off

    Args:
        fname: source file name to build configuration object from (suffix = .cfg)
    """
    def __init__(self, fname=None):
        self.fname = fname
        self.variables = {}

        if self.fname is not None:
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

        Args:
            arg_name: the string name of the argument to retrieve from this config

        Returns:
            the value of the named argument queried
        """
        return self.variables.get(arg_name)

    def hasArg(self, arg_name):
        """
        Check if argument exists (or if it is known by this config object)

        Args:
            arg_name: the string name of the argument to check for the existence of

        Returns:
            True if this config contains this argument, False otherwise
        """
        arg = self.variables.get(arg_name)
        flag = False
        if arg is not None:
            flag = True
        return flag

    def setArg(self, arg_name, arg_value):
        """
        Sets an argument directly

        Args:
            arg_name: the string name of the argument to set within this config

            arg_value: the value name of the argument to set within this config
        """
        self.variables[arg_name] = arg_value
