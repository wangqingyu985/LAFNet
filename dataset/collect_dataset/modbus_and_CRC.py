"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
def reverse(din, len_din):
    din_bin = bin(din)
    din_bin_str = str(din_bin)
    dout_bin_str = ''
    for i in range(len_din):
        if (i < len(din_bin_str)-2):
            dout_bin_str = dout_bin_str + (din_bin_str[len(din_bin_str)-i-1])
        else:
            dout_bin_str = dout_bin_str + '0'
    dout = int(dout_bin_str, 2)
    return dout


def CRC_cal(datas_str='01 06 03 06 00 64'):
    crc16 = 0xffff
    poly = 0x8005
    poly = reverse(poly, 16)
    refin = 1
    refout = 1
    ls = datas_str.split()
    datas = list(ls)
    for data_str in datas:
        data = int(data_str, 16)
        if (refin == 0):
            data = reverse(data, 8)
        crc16 = data ^ crc16
        for i in range(8):
            if 1 & (crc16) == 1:
                crc16 = crc16 >> 1
                crc16 = crc16 ^ poly
            else:
                crc16 = crc16 >> 1
    if (refout == 0):
        result = hex(reverse(crc16, 16))
    else:
        result = hex(crc16)
    return result


def modbus_cal(pressure_dec):
    if pressure_dec > 280:
        return False
    else:
        pressure_hex = hex(pressure_dec)
        if pressure_dec < 16:
            CRC = CRC_cal(datas_str='01 06 03 06 00 0' + pressure_hex[-1])
            result = '01 06 03 06 00 0' \
                     + pressure_hex[-1] + ' ' + CRC[-2] + CRC[-1] + ' ' + CRC[-4] + CRC[-3]
        elif pressure_dec > 255:
            CRC = CRC_cal(datas_str='01 06 03 06 01 ' + pressure_hex[-2] + pressure_hex[-1])
            result = '01 06 03 06 01' \
                     + pressure_hex[-2] + pressure_hex[-1] + ' ' + CRC[-2] + CRC[-1] + ' ' + CRC[-4] + CRC[-3]
        else:
            CRC = CRC_cal(datas_str='01 06 03 06 00 ' + pressure_hex[-2] + pressure_hex[-1])
            if len(CRC) == 6:
                result = '01 06 03 06 00 ' \
                         + pressure_hex[-2] + pressure_hex[-1] + ' ' + CRC[-2] + CRC[-1] + ' ' + CRC[-4] + CRC[-3]
            elif len(CRC) == 5:
                result = '01 06 03 06 00 ' \
                         + pressure_hex[-2] + pressure_hex[-1] + ' ' + CRC[-2] + CRC[-1] + ' ' + '0' + CRC[-3]
            else:
                result = '01 06 03 06 00 ' \
                         + pressure_hex[-2] + pressure_hex[-1] + ' ' + CRC[-2] + CRC[-1] + ' ' + '0' + '0'
        return result
