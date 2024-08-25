import pickle, warnings, time, aiohttp
from pathlib import Path
from datetime import datetime
import warnings, json
from itertools import product
import numpy as np
from yarl import URL
import inspect
from typing import types
import dis

class txt_colors:
    """
    usage: 
    txt_colors.new_style('blue', 1, 34, 1)
    print(txt_colors.blue('txt'))
    
    for color code, see https://ozzmaker.com/add-colour-to-text-in-python/
    """
    
    RESET = '\033[0m' #RESET COLOR

    def new_style(prop_name, style_code, color_code, bg_code):
        def to_style(txt):
            return txt_colors.get_code(style_code, color_code, bg_code) + str(txt) + txt_colors.RESET

        setattr(txt_colors, prop_name, to_style)
                
    def get_code(style_code, color_code, bg_code):
        return  f'\033[{style_code};{color_code};{bg_code}m'
    
    def BOLD_BLUE(txt):
        return txt_colors.get_code(1, 34, 1) + str(txt) + txt_colors.RESET
        
    def BOLD_GREEN(txt):
        return txt_colors.get_code(1, 32, 1) + str(txt) + txt_colors.RESET
        
    def BOLD_RED(txt):
        return txt_colors.get_code(1, 31, 1) + str(txt) + txt_colors.RESET
        
        
    def BOLD_GREY(txt):
        return txt_colors.get_code(1, 37, 1) + str(txt) + txt_colors.RESET
        
    def BOLD(txt):
        return txt_colors.get_code(1, 30, 1) + str(txt) + txt_colors.RESET
   

def show_global_variables(func):
    """
    Usage: 
    @show_global_variables
    def some_function(x):
        return hello
     
    """
    function_globals = func.__globals__
    formatter_otherType = txt_colors.BOLD_RED
    formatter_byType = {
        types.ModuleType:txt_colors.BOLD_GREY,
        types.BuiltinFunctionType:txt_colors.BOLD_GREY, 
        types.FunctionType:txt_colors.BOLD_BLUE, 
        type:txt_colors.BOLD_GREEN
    }
    
        
    print (txt_colors.BOLD(
    f"Global variables of <function '{txt_colors.BOLD_GREEN(func.__qualname__)}'>: "
    ))
    
    
    varnames = _get_global_varnames_recursive(func.__code__)
    
    # no global variables
    if not len(varnames):
        print()
        return func
        
    
    longest_varname = max(max([len(i) for i in varnames]), 30)
    extra_spaces = 20
    

    for each_varname in varnames:
    
        if each_varname in function_globals:
            varType = type(function_globals[each_varname])
            if varType in formatter_byType: 
                formatter = formatter_byType[varType]
            else: formatter = formatter_otherType
        else: 
            try: 
                ## dangerous! 
                varType = type(eval(each_varname, function_globals))
                if varType in [types.BuiltinFunctionType, type]:
                    formatter = formatter_byType[types.BuiltinFunctionType]
                else: 
                    formatter = formatter_otherType
            except:
                varType = "Not yet defined at this point"
                formatter = formatter_otherType
        
            
        print(
        formatter(each_varname), 
        " "*(longest_varname-len(each_varname) + extra_spaces),
        formatter(varType)
        )
    print()
    return func 

def _get_global_varnames_recursive(co):
    
    found  = _get_global_varnames(co)
    
    for x in co.co_consts:
        if hasattr(x, 'co_code'):
            
            found.update(_get_global_varnames_recursive(x))
        
    return found

def _get_global_varnames(func):
    
    globalvars = []
    for instr in (dis.get_instructions(func)):
        if instr.opname in ["LOAD_GLOBAL", "STORE_GLOBAL"]:
            globalvars += [instr.argval, instr.argrepr]
            
    return set(globalvars)
    

def enforce_typing(func):
    """
    TODO: TypeError: isinstance() argument 2 cannot be a parameterized generic
    i don't think this is possible to check

    """
    annotation_byArgName = inspect.signature(func).parameters
    for each_argName in annotation_byArgName:
        if isinstance(annotation_byArgName[each_argName].annotation, types.GenericAlias):
            raise TypeError('Cannot check types for types.GenericAlias')
    

    def wrapped_func(*args, **kwargs):
        annotation_byArgName_local = annotation_byArgName
        empty = inspect.Parameter.empty
        
        # check positional argments
        for each_pos_arg, argName in zip(args, annotation_byArgName_local):
            each_annotation = annotation_byArgName_local[argName].annotation
            if each_annotation is not empty:
                if not isinstance(each_pos_arg, each_annotation):
                    raise TypeError(f"'{argName}' expected {each_annotation}, but received {type(each_pos_arg)}")
           
        # check keyword argments
        for argName, argValue in kwargs.items():
            each_annotation = annotation_byArgName_local[argName].annotation
            if each_annotation is not empty:
                if not isinstance(argValue, each_annotation):
                    raise TypeError(f"'{argName}' expected {each_annotation}, but received {type(argValue)}")
        
        return func(*args, **kwargs)
    return wrapped_func