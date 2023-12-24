import numpy as np
import progressbar
import multiprocessing as mp
from scipy.sparse.csgraph import dijkstra

def shortest_path_worker(todo_queue, output_queue, nbg):
    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        d = dijkstra(nbg, directed=False, indices=index)
        output_queue.put((index, d))


def compute_geodesic_dissimilarity_matrix(nbg):
    dissimilarity_matrix_geodesic = np.zeros((nbg.shape[0], nbg.shape[1]), dtype=np.float32)
    count = 0
    with progressbar.ProgressBar(max_value=nbg.shape[0] ** 2) as bar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in range(nbg.shape[0]):
            todo_queue.put(i)

        # processes = []
        for i in range(mp.cpu_count()):
            todo_queue.put(-1)
            p = mp.Process(target=shortest_path_worker, args=(todo_queue, output_queue, nbg))
            p.start()
            # processes.append(p)
        # for p in processes:
        #     p.join()  # 等待子进程结束

        finished_processes = 0
        while finished_processes != mp.cpu_count():
            i, d = output_queue.get()

            if i == -1:
                finished_processes = finished_processes + 1
            else:
                dissimilarity_matrix_geodesic[i, :] = d
                count = count + len(d)
                bar.update(count)

    return dissimilarity_matrix_geodesic
