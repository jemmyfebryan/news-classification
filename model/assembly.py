from transformers import AutoModel

def reassemble_model(output_file, part_prefix, num_parts):
    with open(output_file, 'wb') as outfile:
        for i in range(num_parts):
            part_file = f"{part_prefix}{chr(97 + i)}"
            with open(part_file, 'rb') as infile:
                outfile.write(infile.read())