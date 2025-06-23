from huggingface_hub import hf_hub_download, list_repo_files, list_repo_tree
from huggingface_hub.hf_api import RepoFile, RepoFolder



class TaskVOS:
    pass


class TaskRVOS:
    pass


class ActionDet:
    pass

class TaskVDE:
    pass

tasks = {
    'AnimalVOS': TaskVOS,
    'AutoVOS':TaskVOS,
    'HumanVOS':TaskVOS,
    'SportsVOS':TaskVOS,

    ## IW
    'IWAnimalVOS':TaskVOS,
    'IWAutoVOS':TaskVOS,
    'IWFurnitureVOS':TaskVOS,
    'IWHumanVOS':TaskVOS,

    ## Street
    'AutoStreetVOS':TaskVOS,
    'BicycleStreetVOS':TaskVOS,
    'HumanStreetVOS':TaskVOS,
    
    ## RVOS
    'AnimalRVOS':TaskRVOS,
    'HumanRVOS':TaskRVOS,

    ## ReVOS,
    'AnimalReVOS':TaskRVOS,
    'AutoReVOS': TaskRVOS,
    'HumanReVOS': TaskRVOS,

    ## CReVOS
    'AnimalCReVOS': TaskRVOS,
    'AutoCReVOS'    : TaskRVOS,
    'HumanCReVOS': TaskRVOS,
    'HumanPartCReVOS': TaskRVOS,
    'EquipmentCReVOS': TaskRVOS,
    # ## VDE
    'StaticVDE': TaskVDE,
    'StreetVDE': TaskVDE,
    'SynVDE': TaskVDE,
    'DynamicVDE': TaskVDE,
}


for task in tasks:
    file_name = 'video/comprehension/' + task + '.zip'
    hf_hub_download(
        repo_id="General-Level/General-Bench-Openset",
        filename=file_name,
        repo_type="dataset",
        local_dir="./data/General-Bench-Openset",
        local_dir_use_symlinks=False
    )


tasks = {
    ## Action Det
    'StaticActionDet': ActionDet,
    'DynamicActionDet': ActionDet,
    'AnimalVG': ActionDet,
    'AutoVG': ActionDet,
    'HumanVG': ActionDet,
}


for task in tasks:
    all_files = list_repo_tree("General-Level/General-Bench-Openset", f"video/comprehension/{task}", repo_type="dataset", recursive=True)
    for file in all_files:
        if isinstance(file, RepoFolder):
            continue
        elif isinstance(file, RepoFile):
            path = file.path
            hf_hub_download(
                repo_id="General-Level/General-Bench-Openset",
                filename=path,
                repo_type="dataset",
                local_dir="./data/General-Bench-Openset",
                local_dir_use_symlinks=False
            )
        else:
            raise ValueError(f"Unknown file type: {file}")
