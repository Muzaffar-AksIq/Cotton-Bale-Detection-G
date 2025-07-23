from datetime import datetime
# shared_state = {
#     "counter": 0,
#     "logs": [],
#     "last_frame": None
# }

shared_state2 = {
    "counter": 0,
    "next_id": 0,
    "tracked": {},
    "history": {},
    "status": {},
    "last_count_time": datetime.now(),
    "logs": [],
    "last_frame": None  # <--- Add this line to store the latest processed frame
}

