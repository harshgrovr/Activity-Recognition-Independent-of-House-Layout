#!/usr/bin/env python
config = {
"learning_rate" : 0.0004,
"num_epochs" : 50,
"decay" : 1e-5,
"input_dim" : 23,
"hidden_dim" : 300,
"layer_dim" : 1,
"output_dim" : 13,
"seq_dim" : 10,
"batch_size" : 16,
"split_ratio" : 0.5,
"num_workers":8,
"accumulation_steps": 3,
"no_of_subset":30 ,
"subset_overlap_length": 2 ,
"resize_width": 224,
"resize_height": 224,
"image_width": 908,
"image_height": 740,
"ActivityIdList" :  [

            {"name": "idle", "id": 0},
            {"name": "leaveHouse", "id": 1},
            {"name": "useToilet", "id": 2},
            {"name": "takeShower", "id": 3},
            {"name": "brushTeeth", "id": 4},
            {"name": "goToBed", "id": 5},
            {"name": "getDressed", "id": 6},
            {"name": "prepareBreakfast", "id": 7},
            {"name": "prepareDinner", "id": 8},
            {"name": "getDrink", "id": 9},
            {"name": "washDishes", "id": 10},
            {"name": "eatDinner", "id": 11},
            {"name": "eatBreakfast", "id": 12}
        ]
}