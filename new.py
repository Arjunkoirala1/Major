def add_extension(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:
        for line in lines:
            # Split the line at '|', add the .png extension to the first part, and then write the new line
            parts = line.strip().split('|')
            image_name = parts[0] + '.png'
            outfile.write(f'{image_name}|{parts[1]}\n')

# Example usage
input_file = r'C:\Users\Nischal\Downloads\segmentation\devnagari.y.txt'  # Replace with the path to your input file
output_file = r'C:\Users\Nischal\Downloads\segmentation\output.txt'  # Replace with the desired output file name

add_extension(input_file, output_file)
