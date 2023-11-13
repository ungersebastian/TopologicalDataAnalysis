# -*- coding: utf-8 -*-
"""
@author: basti
"""

class tda_child(object):
    def __init__(self, parent):
        super(tda_child, self).__init__()
        self.parameter = parent.parameter
        self.parent = parent
        
    
    #######################################################################
    # the following will allow the usage of the _parameter class to store #
    # all the parameters for computation                                  #
    #######################################################################
    
    def __getattr__(self, key):
        if key in list(object.__getattribute__(self, 'parameter').keys()):
            return object.__getattribute__(self, 'parameter')[key]
        else:
            return object.__getattribute__(self, key)
        
    def __setattr__(self, key, value):
        if key == 'parameter':
            object.__setattr__(self, key, value)
        elif key in list(object.__getattribute__(self, 'parameter').keys()):
            self.parameter[key] = value
        else:
            object.__setattr__(self, key, value)
    
    def __hasattr__(self, key):
        if key in list(object.__getattribute__(self, 'parameter').keys()):
            return True
        else:
            return object.__hasattr__(self, key)
    
    def __delattr__(self, key):
        if key in list(object.__getattribute__(self, 'parameter').keys()):
            print('Parameter can´t be removed!')
            return None
        else:
            return object.__delattr__(self, key)
        
        
        
        
    
    
    
    
    
    