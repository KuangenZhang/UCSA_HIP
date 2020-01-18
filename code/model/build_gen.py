import model.source2target as source2target
import model.DNN as DNN
import model.source2target_square as source2target_square

def Generator(source, target, pixelda=False, is_resize = True, 
              dataset = 'NW', sensor_num = 0):  
    if is_resize:
        return source2target_square.Feature(dataset = dataset, 
                           sensor_num = sensor_num)
    else:
        return DNN.Feature()
#        return source2target.Feature()

def Classifier(source, target, is_resize = True, dataset = 'NW'):
    if is_resize:
        return source2target_square.Predictor(dataset = dataset)
    else:
        return DNN.Predictor()
#        return source2target.Predictor()

def DomainClassifier(source, target, is_resize = True, dataset = 'NW'):
    if is_resize:
        return source2target_square.DomainPredictor(dataset = dataset)
    else:
        return DNN.DomainPredictor()
    

