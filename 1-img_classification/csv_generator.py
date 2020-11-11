import os
from datetime import datetime

# -------- CSV output --------
def create_csv(results, results_dir='./results'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')