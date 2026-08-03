# core/producers/
# 事件來源模組。每個 producer 只負責偵測與變化判斷，把 Event 推進 event_queue；
# 「要不要對這個事件說話」一律由 Agent._arbitrate() 決定，不在此處判斷。
