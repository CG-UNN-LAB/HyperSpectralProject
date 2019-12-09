from numba import jit
import numpy as np

@jit(nopython = True)
def chebyshev_metric(vec1, vec2):
    cheb_dist_max = abs(vec1[0] - vec2[0])
    i_cheb_max = 0
    for i in range(1, len(vec1)):
        cheb_dist = abs(vec1[i] - vec2[i])
        if cheb_dist_max < cheb_dist:
            cheb_dist_max = cheb_dist 
            i_cheb_max = i
    return cheb_dist_max

@jit(nopython = True)
def module_numba(vec_1, vec_2):
    '''
    Возвращает длину вектора между двумя точками 
    Параметры
        dot_1 : np.array
            Первая точка
        dot_2 : np.array
            Вторая точка
    Возвращает
        r : float
            Значение длины вектора
    '''
    result = 0
    for i in range(len(vec_1)):
        result = result + pow(vec_1[i] - vec_2[i], 2)
    
    result = pow(result, 0.5)
    return result

@jit(nopython = True)
def pearson_numba(vec_1, vec_2):
    '''
    Возвращает значение корреляции Пирсона между двумя векторами
    Параметры
        vec_1 : np.array
            Вектор один
        vec_2 : np.array
            Вектор два
    Возвращает
        r : float
            Значение корреляции Пирсона между векторами
    '''
    arr_len = vec_1.shape[0]
    xmean = 0
    ymean = 0
    for i in range(arr_len):
        xmean = xmean + vec_1[i]
        ymean = ymean + vec_2[i]
    xmean = xmean / arr_len
    ymean = ymean / arr_len
    xm = vec_1 - xmean
    ym = vec_2 - ymean
    sum_dot_x = 0
    sum_dot_y = 0
    
    for i in range(arr_len):
        sum_dot_x = sum_dot_x + (xm[i] * xm[i])
        sum_dot_y = sum_dot_y + (ym[i] * ym[i])
    normxm = pow(sum_dot_x, 0.5)
    normym = pow(sum_dot_y, 0.5)
    
    r = 0
    for i in range(arr_len):
        r = r + ((xm[i] / normxm) * (ym[i] / normym))
    return r

@jit(nopython = True)
def angle_between_vecs_numba(vec_1, vec_2):
    '''
    Возвращает косинус угла между двумя векторами
    Параметры
        vec_1 : np.array
            Вектор один
        vec_2 : np.array
            Вектор два
    Возвращает
        angle : float
            Косинус угла между векторами 
    '''
    dot = 0
    sqrt_1 = 0
    sqrt_2 = 0
    for i in range(vec_1.shape[0]):
        dot = dot + (vec_1[i] * vec_2[i])
        sqrt_1 = sqrt_1 + (vec_1[i] * vec_1[i])
        sqrt_2 = sqrt_2 + (vec_2[i] * vec_2[i])
    return dot / ( pow(sqrt_1, 0.5) * pow(sqrt_2, 0.5) )

@jit(nopython = True)
def arg_max_compliance(pix, sig_clusters, threshold_clusters, metric):
    '''
    Производит поиск наиболее близкого (коррелирующего) эталона к пикселю,
    с учётом порогового значения
    Параметры
        pix : np.array
            Пиксель, размерностью N
        sig_clusters : np.array
            Массив эталонов, которые соответствуют размерности пикселя pix
        threshold_clusters : np.array
            Массив пороговых значений для каждого эталона
        metric : str
            Определяет метрику для определения схожести между пикселями и эталонами.
    Возвращает
        max_e : float
            Максимальное найденное значение корреляции между пикселями и
            одним из эталонов
        max_i : int
            Номер максимально коррелирующего с пикселем эталона
    '''
    
    extremum_e = 0
    extremum_i = -1
    
    for i in range(len(sig_clusters)):        
        
        if metric == 'angle':
            difference = angle_between_vecs_numba(sig_clusters[i], pix) - threshold_clusters[i]
            
        elif metric == 'pearson':
            difference = pearson_numba(sig_clusters[i], pix) - threshold_clusters[i]
            
        elif metric == 'module':
            difference = threshold_clusters[i] - module_numba(sig_clusters[i], pix)
        
        elif metric == 'chebyshev':
            difference = threshold_clusters[i] - chebyshev_metric(sig_clusters[i], pix)
        
        if extremum_e < difference:
            extremum_e = difference
            extremum_i = i

    return extremum_e, extremum_i

def reference_clustering(HSI, threshold = 0.90, clusters = [], value_mask_on = False, metric = 'angle', rgb_image = []):
    '''
    Кластеризация ГСИ методом накопления эталонов
    Параметры
        HSI : np.array
            ГСИ в виде линейного массива, где само изображение хранится по строкам,
            размер массива (высота изоб. * ширина изоб.) х число каналов
        threshold : float
            Пороговое значение для вхождения пискелей в данный класс.
            Для всех новых (найденных по алгоритму) эталонов будет равняться данному значению.
        clusters : list
            Набор эталонов. Если не указываются - то создаются новые при выполнении данной функции.
            Если указаны несколько эталонов, то они будут участвовать в кластеризации, также
            возможно создание новых эталонов. Представляется в виде списка из трёх массивов:
            [np.array(signatures), np.array(amount_pix_clusts), np.array(thresholds)],
            массив сигнатур эталонов, массив количества пикселей в каждом классе и
            массив пороговых значений для каждого класса.
        value_mask_on : bool
            Нужно ли возвращать построчный массив значений, в виде изображения, отклонений
            пикселей от эталонов своего класса.
        rgb_image : np.array
            Цветное изображение, для раскраски классов, обычно
            представляется в виде RGB синтеза ГСИ. Представляется в виде линейного массива,
            где изображение хранится по строкам. В случае отсутствия RGB изображения для
            раскраски классов будет использоваться заданный набор цветов.
        metrics : str (работа в процессе, необходимо доработать UI)
            Определяет какую метрику будет использовать алгоритм для определения схожести
            между пикселями и эталонами.
                angle - косинус угла
                pearson - корреляция пирсона
                module - 
                chebyshev - 
            
    ''' 
        
    HSI = np.int32( np.array(HSI) )
    rgb_image = np.array(rgb_image)
    
    amo_of_pix = HSI.shape[0]
    len_pix = HSI.shape[1]
    cluster_mask = np.int16( np.zeros(shape = amo_of_pix) )
    cluster_mask_color = np.int16( np.zeros(shape = (amo_of_pix, 3)) )
    if value_mask_on:
        value_mask = np.zeros(shape = amo_of_pix)
            
    if len(clusters) == 0:
        signatures = np.zeros(shape = (1, len_pix))
        amo_of_pix_clusts = np.zeros(shape = 1)
        thresholds = np.zeros(shape = 1)
        cluster_mask[0] = 1
        
        nn_pix = 0
        while all(HSI[nn_pix] != HSI[nn_pix][0]):
            nn_pix = nn_pix + 1
        signatures[0] = HSI[nn_pix]
        
        amo_of_pix_clusts[0] = 1
        thresholds[0] = threshold
        start_clust = 1
        if value_mask_on:
            value_mask[0] = 1
    else:
        
        try:
            assert(len(clusters) == 3)
        except AssertionError:
            print('dimension cluster error')
            print('too many options')
            return
        try:
            assert( len(clusters[0]) == len(clusters[1]) == len(clusters[2]) )
        except AssertionError:
            print('dimension cluster error')
            print('different dimensions of parameters')
            return
        
        signatures = np.array( clusters[0] )
        amo_of_pix_clusts = np.array( clusters[1] )
        thresholds = np.array( clusters[2] )
        start_clust = 0
        amo_of_pix_clusts[:] = 0
    
    #print(amo_of_pix)
    for i in range(start_clust, amo_of_pix):
        
        if all(HSI[i] == HSI[i][0]):
            continue

        ###################################################################################
        max_difference, class_number = arg_max_compliance(HSI[i], signatures, thresholds, metric)
        ###################################################################################
        
        if max_difference > 0:
            cluster_mask[i] = class_number + 1
            amo_of_pix_clusts[class_number] = amo_of_pix_clusts[class_number] + 1
            if value_mask_on:
                value_mask[i] = max_difference + thresholds[class_number]
        else:
            cluster_mask[i] = signatures.shape[0] + 1
            signatures = np.append(signatures, [HSI[i]], axis = 0)
            amo_of_pix_clusts = np.append(amo_of_pix_clusts, 1)
            thresholds = np.append(thresholds, threshold)
            if value_mask_on:
                value_mask[i] = 1.0
                
        #if i % 1000 == 0:
        #    print('\r', end = '')
        #    print(i,  end = '')
    
    if len(rgb_image) != 0:
        for i in range(len(signatures)):
            for col in range(3):
                cluster_mask_color[cluster_mask == i + 1, col] = rgb_image[cluster_mask == i + 1, col].mean()
    else:
        for i in range(len(signatures)):
            cluster_mask_color[cluster_mask == i + 1] = colors_for_clusters[i]
    
    clusters = [signatures, amo_of_pix_clusts, thresholds]

    #print('\n')
    #print("nclass", len(clusters[0]))
    if value_mask_on:
        return cluster_mask, cluster_mask_color, clusters, value_mask
    else:
        return cluster_mask, cluster_mask_color, clusters


def all_stat(hsi, mask, num_clust):
    num_pix_clust = np.nonzero(mask == num_clust)[0]
    mean_sig = np.zeros(shape=hsi.shape[1])
    std_sig = np.zeros(shape=hsi.shape[1])
    min_sig = np.zeros(shape=hsi.shape[1])
    max_sig = np.zeros(shape=hsi.shape[1])

    for i in range(hsi.shape[1]):
        mean_sig[i] = hsi[num_pix_clust, i].mean()
        std_sig[i] = hsi[num_pix_clust, i].std()
        min_sig[i] = hsi[num_pix_clust, i].min()
        max_sig[i] = hsi[num_pix_clust, i].max()

    return mean_sig, std_sig, min_sig, max_sig


def seg_clust_for_mean(hsi, mask, mean, num_clust, intervals, rgb_pic):
    angles_between_men = np.zeros(shape=hsi.shape[0])
    for i, sig in enumerate(hsi):
        angles_between_men[i] = angle_between_vecs_numba(sig, mean)

    plt.figure(figsize=(15, 7))
    h = plt.hist(angles_between_men[mask == num_clust], 1000)
    plt.grid(True)
    plt.xlabel('значения отклонения от среднего', fontsize=18)
    plt.ylabel('количество пикселей с данным отклонением', fontsize=18)

    intervals = np.insert(intervals, 0, h[1][0])
    intervals = np.append(intervals, h[1][-1])
    rgb_color_clust = np.zeros(shape=(intervals.shape[0] - 1, 3))
    for i in range(intervals.shape[0] - 1):
        for c in range(3):
            rgb_color_clust[i, c] = rgb_pic[(angles_between_men >= intervals[i]) &
                                            (angles_between_men < intervals[i + 1]) &
                                            (mask == num_clust), c].mean()
    rgb_color_clust = np.int16(rgb_color_clust * 2)

    col_norm = rgb_color_clust / 255
    plt.plot([intervals[0], intervals[0]], [0, h[0].max()],
             color=col_norm[0], linewidth=4)
    plt.text(intervals[0], h[0].max(), np.float16(intervals[0]))
    plt.plot([intervals[-1], intervals[-1]],
             [0, h[0].max()], color='black', linewidth=4)
    plt.text(intervals[-1], h[0].max(), np.float16(intervals[-1]))

    for i in range(1, intervals.shape[0] - 1):
        col = np.array(rgb_color_clust[i]) / 255
        plt.plot([intervals[i], intervals[i]], [0, h[0].max()],
                 color=col_norm[i], linewidth=4)
        plt.text(intervals[i], h[0].max(), intervals[i])

    pic_sep_color_clust = np.zeros(shape=(mask.shape[0], 3))

    for i in range(intervals.shape[0] - 1):
        pic_sep_color_clust[(angles_between_men >= intervals[i]) & (
            angles_between_men < intervals[i + 1]) & (mask == num_clust)] = rgb_color_clust[i]
        num_of_pix = pic_sep_color_clust[(angles_between_men >= intervals[i]) & (
            angles_between_men < intervals[i + 1]) & (mask == num_clust)].shape[0]
        plt.text(intervals[i], h[0].max() - (h[0].max()
                                             * 0.08), str(num_of_pix) + '\n пикселей')

    fh.show_1D_img(pic_sep_color_clust, 1924, 753,
                   _show_x=30, _show_y=40, flip=True)
    return pic_sep_color_clust


def seg_clust(hsi, mask, mask_v, num_clust, intervals, rgb_pic):
    clust_values = mask_v[mask == num_clust]
    plt.figure(figsize=(15, 7))
    h = plt.hist(clust_values, 1000)
    plt.grid(True)
    plt.xlabel('значения отклонения от среднего', fontsize=18)
    plt.ylabel('количество пикселей с данным отклонением', fontsize=18)

    intervals = np.insert(intervals, 0, h[1][0])
    intervals = np.append(intervals, h[1][-1])
    rgb_color_clust = np.zeros(shape=(intervals.shape[0] - 1, 3))
    for i in range(intervals.shape[0] - 1):
        rgb_color_clust[i, 0] = rgb_pic[(mask_v >= intervals[i]) & (
            mask_v < intervals[i + 1]) & (mask == num_clust), 0].mean()
        rgb_color_clust[i, 1] = rgb_pic[(mask_v >= intervals[i]) & (
            mask_v < intervals[i + 1]) & (mask == num_clust), 1].mean()
        rgb_color_clust[i, 2] = rgb_pic[(mask_v >= intervals[i]) & (
            mask_v < intervals[i + 1]) & (mask == num_clust), 2].mean()
    rgb_color_clust = np.int16(rgb_color_clust * 2)

    col_norm = rgb_color_clust / 255
    plt.plot([intervals[0], intervals[0]], [0, h[0].max()],
             color=col_norm[0], linewidth=4)
    plt.text(intervals[0], h[0].max(), np.float16(intervals[0]))
    plt.plot([intervals[-1], intervals[-1]],
             [0, h[0].max()], color='black', linewidth=4)
    plt.text(intervals[-1], h[0].max(), np.float16(intervals[-1]))

    for i in range(1, intervals.shape[0] - 1):
        col = np.array(rgb_color_clust[i]) / 255
        plt.plot([intervals[i], intervals[i]], [0, h[0].max()],
                 color=col_norm[i], linewidth=4)
        plt.text(intervals[i], h[0].max(), intervals[i])

    pic_sep_color_clust = np.zeros(shape=(mask.shape[0], 3))

    for i in range(intervals.shape[0] - 1):
        pic_sep_color_clust[(mask_v >= intervals[i]) & (
            mask_v < intervals[i + 1]) & (mask == num_clust)] = rgb_color_clust[i]
        num_of_pix = pic_sep_color_clust[(mask_v >= intervals[i]) & (
            mask_v < intervals[i + 1]) & (mask == num_clust)].shape[0]
        plt.text(intervals[i], h[0].max() - (h[0].max()
                                             * 0.08), str(num_of_pix) + '\n пикселей')

    fh.show_1D_img(pic_sep_color_clust, 1924, 753,
                   _show_x=30, _show_y=40, flip=True)
    return pic_sep_color_clust
	

colors_for_clusters = (
	(199, 193, 191) , (94, 188, 209) , (61, 79, 68) , (126, 100, 5) , (2, 104, 78) , (150, 43, 117) , (141, 133, 70) , (150, 149, 197) ,
	(147, 19, 84) , (81, 160, 88) , (164, 91, 2) , (29, 23, 2) , (226, 0, 39) , (231, 171, 99) , (76, 96, 1) , (156, 105, 102) ,
	(100, 84, 123) , (151, 151, 158) , (0, 106, 102) , (57, 20, 6) , (244, 215, 73) , (0, 69, 210) , (0, 108, 49) , (221, 182, 208) ,
	(124, 101, 113) , (159, 178, 164) , (0, 216, 145) , (21, 160, 138) , (188, 101, 233) , (255, 26, 89) , (198, 220, 153) , (32, 59, 60) ,
	(103, 17, 144) , (107, 58, 100) , (245, 225, 255) , (255, 160, 242) , (204, 170, 53) , (55, 69, 39) , (139, 180, 0) , (121, 120, 104) ,
	(198, 0, 90) , (59, 0, 10) , (200, 98, 64) , (41, 96, 124) , (64, 35, 52) , (125, 90, 68) , (204, 184, 124) , (184, 129, 131) ,
	(170, 81, 153) , (181, 214, 195) , (163, 132, 105) , (159, 148, 240) , (167, 69, 113) , (184, 148, 166) , (113, 187, 140) , (0, 180, 51) ,
	(121, 0, 215) , (167, 117, 0) , (99, 103, 169) , (160, 88, 55) , (107, 0, 44) , (119, 38, 0) , (215, 144, 255) , (155, 151, 0) ,
	(189, 201, 210) , (159, 160, 100) , (190, 71, 0) , (101, 129, 136) , (131, 164, 133) , (69, 60, 35) , (71, 103, 93) , (58, 63, 0) ,
	(6, 18, 3) , (223, 251, 113) , (134, 142, 126) , (152, 208, 88) , (108, 143, 125) , (215, 191, 194) , (60, 62, 110) , (216, 61, 102) ,
	(255, 255, 0) , (28, 230, 255) , (255, 52, 255) , (255, 74, 70) , (0, 137, 65) , (0, 111, 166) , (163, 0, 89) , (252, 0, 156) ,
	(255, 219, 229) , (122, 73, 0) , (0, 0, 166) , (99, 255, 172) , (183, 151, 98) , (0, 77, 67) , (143, 176, 255) , (153, 125, 135) ,
	(90, 0, 7) , (128, 150, 147) , (254, 255, 230) , (27, 68, 0) , (79, 198, 1) , (59, 93, 255) , (74, 59, 83) , (255, 47, 128) ,
	(97, 97, 90) , (186, 9, 0) , (107, 121, 0) , (0, 194, 160) , (255, 170, 146) , (255, 144, 201) , (185, 3, 170) , (209, 97, 0) ,
	(221, 239, 255) , (0, 0, 53) , (123, 79, 75) , (161, 194, 153) , (48, 0, 24) , (10, 166, 216) , (1, 51, 73) , (0, 132, 111) ,
	(55, 33, 1) , (255, 181, 0) , (194, 255, 237) , (160, 121, 191) , (204, 7, 68) , (192, 185, 178) , (194, 255, 153) , (0, 30, 9) ,
	(84, 158, 121) , (255, 246, 159) , (32, 22, 37) , (114, 65, 143) , (188, 35, 255) , (153, 173, 192) , (58, 36, 101) , (146, 35, 41) ,
	(91, 69, 52) , (253, 232, 220) , (64, 78, 85) , (0, 137, 163) , (203, 126, 152) , (164, 232, 4) , (50, 78, 114) , (106, 58, 76) ,
	(131, 171, 88) , (0, 28, 30) , (209, 247, 206) , (0, 75, 40) , (200, 208, 246) , (163, 164, 137) , (128, 108, 102) , (34, 40, 0) ,
	(191, 86, 80) , (232, 48, 0) , (102, 121, 109) , (218, 0, 124) , (255, 255, 254) , (138, 219, 180) , (30, 2, 0) , (91, 78, 81) ,
	(200, 149, 197) , (50, 0, 51) , (255, 104, 50) , (102, 225, 211) , (207, 205, 172) , (208, 172, 148) , (126, 211, 121) , (1, 44, 88) ,
	(120, 158, 201) , (109, 128, 186) , (0, 166, 170) , (94, 255, 3) , (228, 255, 252) , (27, 225, 119) , (188, 177, 229) , (118, 145, 47) ,
	(0, 49, 9) , (0, 96, 205) , (210, 0, 150) , (137, 85, 99) , (41, 32, 29) , (91, 50, 19) , (167, 111, 66) , (137, 65, 46) ,
	(26, 58, 42) , (73, 75, 90) , (168, 140, 133) , (244, 171, 170) , (163, 243, 171) , (0, 198, 200) , (234, 139, 102) , (149, 138, 159) ,
	(0, 72, 156) , (111, 0, 98) , (12, 189, 102) , (238, 195, 255) , (69, 109, 117) , (183, 123, 104) , (122, 135, 161) , (120, 141, 102) ,
	(122, 123, 255) , (214, 142, 1) , (53, 51, 57) , (120, 175, 161) , (254, 178, 198) , (117, 121, 124) , (131, 115, 147) , (148, 58, 77) ,
	(181, 244, 255) , (210, 220, 213) , (149, 86, 189) , (106, 113, 74) , (0, 19, 37) , (2, 82, 95) , (10, 163, 247) , (233, 129, 118) ,
	(47, 93, 155) , (108, 94, 70) , (210, 91, 136) , (91, 101, 108) , (0, 181, 127) , (84, 92, 70) , (134, 96, 151) , (54, 93, 37) ,
	(136, 85, 120) , (250, 208, 159) , (255, 138, 154) , (209, 87, 160) , (190, 196, 89) , (69, 102, 72) , (0, 134, 237) , (136, 111, 76) ,
	(52, 54, 45) , (180, 168, 189) , (149, 63, 0) , (69, 44, 44) , (99, 99, 117) , (163, 200, 201) , (255, 145, 63) , (147, 138, 129) ,
	(87, 83, 41) , (0, 254, 207) , (176, 91, 111) , (140, 208, 255) , (59, 151, 0) , (4, 247, 87) , (200, 161, 161) , (30, 110, 0) ,
	(231, 115, 206) , (216, 106, 120) , (62, 137, 190) , (202, 131, 78) , (81, 138, 135) , (91, 17, 60) , (85, 129, 59) , (231, 4, 196) ,
	(0, 0, 95) , (169, 115, 153) , (75, 129, 96) , (89, 115, 138) , (255, 93, 167) , (247, 201, 191) , (100, 49, 39) , (81, 58, 1) ,
	(37, 47, 153) , (0, 204, 255) , (103, 78, 96) , (146, 137, 107))

