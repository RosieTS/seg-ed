import os
import numpy as np
import epi_hover_merge as ep

def save_random_image(image_file_name, json_file_path):

    tile_id = ep.get_basename(rand_image_file)
    json_file_name = ep.get_json_name(tile_id, json_file_path)
    temp_file = os.path.join('tmp', tile_id + '.npy')

    if not os.path.exists(json_file_name):
        print(f"File {json_file_name} does not exist.")
        return
                 
    epi_mask = ep.open_and_rescale_prediction(temp_file)
    epi_nuc_uids, epi_nuc_centroids, epi_nuc_contours = ep.get_epithelium_nuclei(json_file_name, epi_mask)
            
    if len(epi_nuc_uids) == 0:
        print(f"No epithelial nuclei identified in file {json_file_name}.")
        return

    ep.save_sample_image(rand_image_file, json_file_path, tile_id, epi_mask, epi_nuc_contours)


if __name__ == "__main__":

    command_line_args = ep.parse_command_line_args()
    json_file_path = command_line_args.json_path

    image_file_names = ep.get_image_file_names(command_line_args.image_path)

    rand_image_file_names = np.random.choice(image_file_names, size = 20).tolist()
    print(rand_image_file_names)

    ep.run_model_for_predictions(command_line_args.model_path, rand_image_file_names, 
        command_line_args.bs, command_line_args.lw)

    for rand_image_file in rand_image_file_names:

        #rand_image_file = np.random.choice(image_file_names)
        save_random_image(rand_image_file, json_file_path)
            
      





    #epi_nuc_data = loop_through_tiles(image_file_names, command_line_args.json_path)

    #print("Head and tail of final dataframe:")
    #print(epi_nuc_data.head())
    #print(epi_nuc_data.tail())