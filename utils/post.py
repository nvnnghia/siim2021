# collect the log and models

def test_colab():
    '''
    Check if the code is running at colab.

    '''
    try:
        import google.colab
        return True
    except:
        return False

