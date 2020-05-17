#!/usr/bin/env python

config = {
"learning_rate" : 1e-3,
"num_epochs" : 100,
"decay" : 1e-6,
"input_dim" : 23,
"hidden_dim" : 100,
"layer_dim" : 2,
"output_dim" : 18,
"seq_dim" : 5,
"batch_size" : 8,
"split_ratio" : 2/3,
"num_workers":8,
"accumulation_steps": 20,
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

