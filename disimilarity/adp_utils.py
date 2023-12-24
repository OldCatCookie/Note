import numpy as np
import progressbar
import multiprocessing as mp

def adp_dissimilarities_worker(todo_queue, output_queue, csi_time_domain):
    def adp_dissimilarities(index):
        h = csi_time_domain[index,:,:,:]
        w = csi_time_domain[index:,:,:,:]

        dotproducts = np.abs(np.einsum("bmt,lbmt->lbt", np.conj(h), w, optimize="optimal"))**2
        norms = np.real(np.einsum("bmt,bmt->bt", h, np.conj(h), optimize="optimal") * np.einsum("lbmt,lbmt->lbt", w, np.conj(w), optimize="optimal"))

        return np.sum(1 - dotproducts / norms, axis=(1, 2))

    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        output_queue.put((index, adp_dissimilarities(index)))


def compute_adp_dissimilarity_matrix(csi_time_domain):
    adp_dissimilarity_matrix = np.zeros((csi_time_domain.shape[0], csi_time_domain.shape[0]), dtype=np.float32)
    count = 0
    with progressbar.ProgressBar(max_value=csi_time_domain.shape[0] ** 2) as bar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in range(csi_time_domain.shape[0]):
            todo_queue.put(i)

        # processes = []
        for i in range(mp.cpu_count()):
            todo_queue.put(-1)
            p = mp.Process(target=adp_dissimilarities_worker, args=(todo_queue, output_queue, csi_time_domain))
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
                adp_dissimilarity_matrix[i, i:] = d
                adp_dissimilarity_matrix[i:, i] = d
                count = count + 2 * len(d) - 1
                bar.update(count)

    return adp_dissimilarity_matrix
