"""
Script for training and evaluation sota models.
Author: ChunWei Shen
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="MTGNN")
parser.add_argument("--data",type=str,default="metr_la")
args = parser.parse_args()
model = args.model
data = args.data

def main():
    if model == "DCRNN":
        if data == "metr_la":
            cmd = "python -m tools.main model=DCRNN data=metr_la trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[20, 30, 40, 50]' trainer.lr_skd.gamma=0.1 trainer.optimizer.lr=0.01\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 data.dp.priori_gs.type=dual_random_walk"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=DCRNN data=pems_bay trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[20, 30, 40, 50]' trainer.lr_skd.gamma=0.1 trainer.optimizer.lr=0.01\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 data.dp.priori_gs.type=dual_random_walk"
        elif data == "pems03":
            cmd = "python -m tools.main model=DCRNN data=pems03 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 data.dp.priori_gs.type=dual_random_walk\
                trainer.dataloader.batch_size=32"
        elif data == "pems04":
            cmd = "python -m tools.main model=DCRNN data=pems04 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 data.dp.priori_gs.type=dual_random_walk\
                trainer.dataloader.batch_size=32"
        elif data == "pems07":
            cmd = "python -m tools.main model=DCRNN data=pems07 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 data.dp.priori_gs.type=dual_random_walk\
                trainer.dataloader.batch_size=32"
        elif data == "pems08":
            cmd = "python -m tools.main model=DCRNN data=pems07 trainer/lr_skd=multistep\
                'trainer.lr_skd.milestones=[80]' trainer.lr_skd.gamma=0.3 trainer.optimizer.lr=0.003\
                trainer.optimizer.eps=1e-3 trainer.optimizer.weight_decay=0 data.dp.priori_gs.type=dual_random_walk"
    elif model == "STGCN":
        if data == "metr_la":
            cmd = "python -m tools.main model=STGCN data=metr_la trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STGCN data=pems_bay trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=STGCN data=pems03 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=STGCN data=pems04 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=STGCN data=pems07 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=STGCN data=pems08 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                data.dp.time_enc.add_tid=False data.dp.priori_gs.type=laplacian\
                model.model_params.st_params.n_series=170"
    elif model == "GWNet":
        if data == "metr_la":
            cmd = "python -m tools.main model=GWNet data=metr_la trainer.lr_skd=null model.model_params.n_series=207\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=GWNet data=pems_bay trainer.lr_skd=null model.model_params.n_series=325\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems03":
            cmd = "python -m tools.main model=GWNet data=pems03 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=358\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems04":
            cmd = "python -m tools.main model=GWNet data=pems04 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=307\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems07":
            cmd = "python -m tools.main model=GWNet data=pems07 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=883\
                data.dp.priori_gs.type=dbl_transition"
        elif data == "pems08":
            cmd = "python -m tools.main model=GWNet data=pems08 trainer/lr_skd=multistep trainer.optimizer.lr=0.002\
                'trainer.lr_skd.milestones=[1, 50]' model.model_params.n_series=170\
                data.dp.priori_gs.type=dbl_transition"
    elif model == "MTGNN":
        if data == "metr_la":
            cmd = "python -m tools.main model=MTGNN data=metr_la trainer.lr_skd=null +trainer.cl.lv_up_period=2500\
                +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=MTGNN data=pems_bay trainer.lr_skd=null +trainer.cl.lv_up_period=2500\
                +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=MTGNN data=pems03 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1500 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=MTGNN data=pems04 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1000 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=MTGNN data=pems07 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1500 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=MTGNN data=pems08 trainer.lr_skd=null trainer.dataloader.batch_size=32\
                +trainer.cl.lv_up_period=1000 +trainer.cl.task_lv_max=12 model.model_params.gsl_params.n_series=170"
    elif model == "STSGCN":
        if data == "metr_la":
            cmd = "python -m tools.main model=STSGCN data=metr_la trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                 data.dp.priori_gs.type=binary model.model_params.st_params.n_series=207"
        elif data == "pems_bay":
            cmd = "python -m tools.main model=STSGCN data=pems_bay trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                 data.dp.priori_gs.type=binary model.model_params.st_params.n_series=325"
        elif data == "pems03":
            cmd = "python -m tools.main model=STSGCN data=pems03 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=16 data.dp.time_enc.add_tid=False\
                 data.dp.priori_gs.type=binary model.model_params.st_params.n_series=358"
        elif data == "pems04":
            cmd = "python -m tools.main model=STSGCN data=pems04 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                 data.dp.priori_gs.type=binary model.model_params.st_params.n_series=307"
        elif data == "pems07":
            cmd = "python -m tools.main model=STSGCN data=pems07 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=16 data.dp.time_enc.add_tid=False\
                 data.dp.priori_gs.type=binary model.model_params.st_params.n_series=883"
        elif data == "pems08":
            cmd = "python -m tools.main model=STSGCN data=pems08 trainer/lr_skd=multistep trainer.es.patience=30\
                'trainer.lr_skd.milestones=[50, 80]' trainer.dataloader.batch_size=32 data.dp.time_enc.add_tid=False\
                 data.dp.priori_gs.type=binary model.model_params.st_params.n_series=170"

    os.system(cmd)

if __name__ == "__main__":
    main()