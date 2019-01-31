CKPT:=none
PB:=mobilenet_v1_0.5_240_320_frozen.pb
OUTNODE:=Flowers/Final/BiasAdd


train:
	python3 tf_train.py
freeze:
	python3 freeze_graph.py ${CKPT} ${PB} ${OUTNODE}  && \
	cp -f ${PB} ~/Documents/kendryte-model-compiler/pb_files/ 
convert:
	cd ~/Documents/kendryte-model-compiler/ && \
	python3 __main__.py --dataset_input_name "inputs:0" \
                    --dataset_loader "dataset_loader/img_neg1_1.py" \
                    --image_h 240 --image_w 320 \
                    --dataset_pic_path "dataset/flowers" \
                    --model_loader "model_loader/pb" \
                    --pb_path "pb_files/${PB}" \
                    --tensor_output_name "${OUTNODE}"
	


