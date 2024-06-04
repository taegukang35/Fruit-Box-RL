import numpy as np

def sum_boxes(board):
    boxes = dict()
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if (i-1, j) in boxes:
                boxes[(i, j)] = boxes[(i-1, j)] + np.sum(board[i, :j])
            elif (i, j-1) in boxes:
                boxes[(i, j)] = boxes[(i, j-1)] + np.sum(board[:i, j])
            else:
                boxes[(i, j)] = board[i, j]
    return boxes

board = np.random.randint(1, 10, (9, 18))

boxes = sum_boxes(board)
print(len(boxes))
print('hehe')