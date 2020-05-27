#!/usr/bin/env python
config = {
"learning_rate" : 0.0025,
"num_epochs" : 70,
"decay" : 1e-5,
"input_dim" : 13,
"hidden_dim" : 32,
"layer_dim" : 1,
"output_dim" : 11,
"seq_dim" : 10,
"batch_size" : 64,
"split_ratio" : 2/3,
"num_workers":8,
"accumulation_steps": 2,
"no_of_subset":30 ,
"subset_overlap_length": 2 ,
"resize_width": 224 ,
"resize_height": 224 ,
"image_width": 908 ,
"image_height": 740 ,
"ActivityIdList" :  [

    {"name": "idle", "id": 0},
    {"name": "goToBed", "id": 1},
    {"name": "useToilet", "id": 2},
    {"name": "takeShower", "id": 3},
    {"name": "prepareBreakfast", "id": 4},
    {"name": "prepareLunch", "id": 5},
    {"name": "grooming", "id": 6},
    {"name": "watchTV", "id": 7},
    {"name": "leaveHouse", "id": 8},
    {"name": "getSnack", "id": 9},
    {"name": "prepareDinner", "id": 10}
]
}