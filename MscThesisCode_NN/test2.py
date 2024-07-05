import numpy as np
def map_range(values, input_start, input_end, output_start, output_end):

    mapped_values = []
    for value in values:
        # Calculate the proportion of value in the input range
        proportion = (value - input_start) / (input_end - input_start)
        
        # Map the proportion to the output range
        mapped_value = output_start + (proportion * (output_end - output_start))
        np.array(mapped_values.append(int(mapped_value)))
    
    return mapped_values

# Example usage:
input_values = [15,25,35]  # Values in the range (0, 50)
mapped_values = map_range(input_values, 0, 53, 0, 20)
print(mapped_values)