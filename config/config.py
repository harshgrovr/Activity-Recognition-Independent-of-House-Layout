#!/usr/bin/env python

config = {
"learning_rate" : 0.0075,
"num_epochs" : 150,
"decay" : 1e-6,
"input_dim" : 13,
"hidden_dim" : 64,
"layer_dim" : 1,
"output_dim" : 11,
"seq_dim" : 10,
"batch_size" : 128,
"split_ratio" : 2/3,
"num_workers":8,
"accumulation_steps": 1,
"no_of_subset":30 ,
"subset_overlap_length": 2 ,
"resize_width": 224 ,
"resize_height": 224 ,
"image_width": 908 ,
"image_height": 740 ,
"ActivityIdList" :  [

            {"name": "idle", "id": 0},
            {"name": "leaveHouse", "id": 1},
            {"name": "useToilet", "id": 2},
            {"name": "takeShower", "id": 3},
            {"name": "goToBed", "id": 4},
            {"name": "prepareBreakfast", "id": 5},
            {"name": "prepareDinner", "id": 6},
            {"name": "getSnack", "id": 7},
            {"name": "prepareLunch", "id": 8},
            {"name": "grooming", "id": 9},
            {"name": "watchTV", "id": 10}
        ]
}
