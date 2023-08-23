from rd_rl.raptor.init_raptor import init_matlab, init_sparc_rd

if __name__ == "__main__":
    raptor_repo_root = "/home/awang/raptor"
    eng_handle = init_matlab(raptor_repo_root)
    Ip = init_sparc_rd(raptor_repo_root, eng_handle)
