
# define maximum value
max_pH = 14.0
max_temp = 60.0
max_conc = 1000.0
max_mgcl2 = 50.0
max_nacl = 2000.0
max_kcl = 1000.0
max_cagent = 50.0
max_glyc = 10.0

# define crowding agents
cagent_list = ['PEG300-1k', 'PEG3k-6k', 'PEG8k-20k', 'Ficoll', 'Dextran -40', 'Dextran 70-']
cond_list = ['temp', 'conc', 'pH'] + cagent_list + ['mgcl2', 'nacl', 'kcl'] + ['glyc']

# define amino acids
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# base condition for screening and mutation analysis
base_cond_screen = {
    'temp': 25.0,   # degree Celsius
    'conc': 100.0,   # uM
    'pH': 7.3,
    
    'PEG300-1k': 0.0,   # %
    'PEG3k-6k': 0.0,   # %
    'PEG8k-20k': 0.0,   # %
    'Ficoll': 0.0,   # %
    'Dextran -40': 0.0,   # %
    'Dextran 70-': 0.0,   # %
    
    'mgcl2': 0.0,   # mM
    'nacl': 160.0,   # mM
    'kcl': 0.0,   # mM

    'glyc': 0.0  # %
}

base_cond_mutation = {
    'temp': 25.0,   # degree Celsius
    'conc': 100.0,   # uM
    'pH': 7.3,
    
    'PEG300-1k': 0.0,   # %
    'PEG3k-6k': 0.0,   # %
    'PEG8k-20k': 0.0,   # %
    'Ficoll': 0.0,   # %
    'Dextran -40': 0.0,   # %
    'Dextran 70-': 0.0,   # %
    
    'mgcl2': 0.0,   # mM
    'nacl': 160.0,   # mM
    'kcl': 0.0,   # mM
    
    'glyc': 0.0  # %
}


base_cond_screen_multiple = {
    'temp': 37.0,   # degree Celsius
    'conc': 10.0,   # uM
    'pH': 7.4,
    
    'PEG300-1k': 0.0,   # %
    'PEG3k-6k': 0.0,   # %
    'PEG8k-20k': 0.0,   # %
    'Ficoll': 0.0,   # %
    'Dextran -40': 0.0,   # %
    'Dextran 70-': 0.0,   # %
        
    'mgcl2': 0.0,
    'nacl': 137.0,
    'kcl': 3.0,
    
    'glyc': 0.0
}

# store maximum values in a dictionary
max_values = {
    'temp': max_temp,
    'conc': max_conc,
    'pH': max_pH,
    'mgcl2': max_mgcl2,
    'nacl': max_nacl,
    'kcl': max_kcl,
    'salt': max_nacl,
    'cagent': max_cagent,
    'glyc': max_glyc
}

# define scaling and biasing factors for condition normalization
scale_cond = {
    'temp': 1.0,
    'conc': 0.1,
    'salt': 20,
    'pH': 0.1
}
bias_cond = {
    'temp': 0.0,
    'conc': 0.0,
    'salt': 0.0,
    'pH': 5.0
}


def get_cond_index(cond_name):
    """ Get the index of a condition in the condition vector.
    
    Args:
        cond_name (str): Name of the condition.
    """
    return cond_list.index(cond_name)


def normalize_condition(cond_dict):
    """ Normalize the condition dictionary using predefined maximum values.
    
    Args:
        cond_dict (dict): Dictionary containing condition values.
        
    Returns:
        dict: Normalized condition dictionary.
    """
    normalized_dict = {}
    for key, value in cond_dict.items():
        if key in max_values:
            normalized_dict[key] = value / max_values[key]
        elif key in cagent_list:
            normalized_dict[key] = value / max_cagent
        else:
            normalized_dict[key] = value  # If no max value defined, keep original
    return normalized_dict


def denormalize_condition(cond_dict):
    """ Denormalize the condition dictionary using predefined maximum values.
    
    Args:
        cond_dict (dict): Dictionary containing normalized condition values.
        
    Returns:
        dict: Denormalized condition dictionary.
    """
    denormalized_dict = {}
    for key, value in cond_dict.items():
        if key in max_values:
            denormalized_dict[key] = value * max_values[key]
        elif key in cagent_list:
            denormalized_dict[key] = value * max_cagent
        else:
            denormalized_dict[key] = value  # If no max value defined, keep original
    return denormalized_dict
