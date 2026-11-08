# Constant Variables

# -----------------------
# DATASET ROOT DIRS
# -----------------------
data_root = "../datasets"


# augmentation
augmentation_mapping = {"svhn":    [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                   ["Rotate"], 
                                   ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],

                        "cifar10": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                    ["Rotate"], 
                                    ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],

                        "cifar100": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                     ["Rotate"], 
                                     ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],

                        "tinyimagenet": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                         ["Rotate"], 
                                         ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],
                                         
                        "cub": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                ["Rotate"], 
                                ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],
                        
                        "aircraft": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                     ["Rotate"], 
                                     ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]]}


temperature1_scheduling_mapping = {"mnist": [1.0, 0.5, 0.05],  
                                  "svhn": [0.1, 0.05, 0.01, 0.005],
                                  "cifar10": [0.1, 0.01, 0.005],  #[0.05, 0.01, 0.005],     #[0.05, 0.049, 0.048, 0.047, 0.046, 0.045, 0.044, 0.043, 0.042, 0.041, 0.04, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031],
                                  "cifar-10-100-10": [0.1, 0.05, 0.01],
                                  "cifar-10-100-50": [0.1, 0.05, 0.01],
                                  "tinyimagenet": [0.1, 0.05, 0.01]}

temperature2_scheduling_mapping = {"mnist": [1.0, 0.5, 0.05],  
                                  "svhn": [0.1, 0.05, 0.01, 0.005],
                                  "cifar10": [1., 1., 1.], #[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                  "cifar-10-100-10": [0.1, 0.05, 0.01],
                                  "cifar-10-100-50": [0.1, 0.05, 0.01],
                                  "tinyimagenet": [0.1, 0.05, 0.01]}



temperature_scheduling_epoch_mapping = {"mnist": [0, 50, 150],  
                                        "svhn": [0, 100, 200, 300],
                                        "cifar10": [0, 10, 20],      #[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200],
                                        "cifar-10-100-10": [0, 100, 200],
                                        "cifar-10-100-50": [0, 100, 200],
                                        "tinyimagenet": [0, 100, 200]}

sampling_scheduling_epoch_mapping = {"mnist": [200],  
                                     "svhn": [200],
                                     "cifar10": [200],
                                     "cifar-10-100-10": [200],
                                     "cifar-10-100-50": [200],
                                     "tinyimagenet": [200]}



#[0.1, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005]
#[1.1, 1.11, 1.12,  1.13,  1.14, 1.15, 1.16, 1.17,  1.18,  1.19,  1.2,  1.21,  1.22, 1.21,  1.2,  1.19,  1.18, 1.17,  1.16, 1.15]
#[0,  10,    20,    30,    40,    50,   60,   70,   80,    90,    100,  110, 120,     130,  140,  150,   160,  170,  180  200]
