#!/usr/bin/env python

config = {
"learning_rate" : 0.0025,
"num_epochs" : 50,
"decay" : 0.000015,
"input_dim" : 23,
"hidden_dim" : 32,
"layer_dim" : 1,
"output_dim" : 18,
"seq_dim" : 10,
"batch_size" : 128,
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
            {"name": "leaveHouse", "id": 1},
            {"name": "useToilet", "id": 2},
            {"name": "takeShower", "id": 3},
            {"name": "brushTeeth", "id": 4},
            {"name": "goToBed", "id": 5},
            {"name": "getDressed", "id": 6},
            {"name": "prepareBreakfast", "id": 7},
            {"name": "prepareDinner", "id": 8},
            {"name": "getSnack", "id": 9},
            {"name": "getDrink", "id": 10},
            {"name": "loadDishwasher", "id": 11},
            {"name": "unloadDishwasher", "id": 12},
            {"name": "storeGroceries", "id": 13},
            {"name": "washDishes", "id": 14},
            {"name": "answerPhone", "id": 15},
            {"name": "eatDinner", "id": 16},
            {"name": "eatBreakfast", "id": 17}
        ]
}
