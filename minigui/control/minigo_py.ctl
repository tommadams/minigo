board_size = 19

players = {
  "minigo_py" : Player("python"
                       " -u"
                       " gtp.py"
                       " --load_file=saved_models/000865-bulwark"
                       " --minigui_mode=true"
                       " --num_readouts=128"
                       " --conv_width=256"
                       " --resign_threshold=-0.8"
                       " --verbose=2",
                       startup_gtp_commands=[],
                       environ={"BOARD_SIZE": str(board_size)}),
}
