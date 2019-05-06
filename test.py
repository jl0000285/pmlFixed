import dbasehandler as dbh
handler = dbh.DbHandler()
handler.setup_session()
#handler.dbInit()
#handler.populate_runs_all()
#handler.guesses_exhaustive()
handler.populate_learning_curves()
#handler.guesses_sampling()
#handler.populate_results()
