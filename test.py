import dbasehandler as dbh
handler = dbh.DbHandler()
handler.setup_session()
#handler.guesses_exhaustive()
handler.populate_learning_curves()
