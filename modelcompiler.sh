# !/bin/bash
python3 /media/zqh/ProJects/Kendryte_K210/kendryte-model-compiler/__main__.py \
                    --dataset_input_name "Input_image:0" \
                    --tensor_output_name "Output_label:0" \
                    --eight_bit_mode True \
                    --model_loader "/media/zqh/ProJects/Kendryte_K210/kendryte-model-compiler/model_loader/pb" \
                    --dataset_pic_path "/media/zqh/ProJects/Kendryte_K210/kendryte-model-compiler/dataset/yolo_240_320" \
                    --dataset_loader "/media/zqh/ProJects/Kendryte_K210/kendryte-model-compiler/dataset_loader/img_0_1.py" \
                    --image_h 240 \
                    --image_w 320 \
                    --pb_path "log/train/save_18:40:14/saved_model.pb" 