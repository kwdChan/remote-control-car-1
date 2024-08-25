import machine
import utime

analog_value = machine.ADC(26)
conversion_factor = 3.3 / (65535)
x = [0]*100

t0 = utime.ticks_cpu()

while True:
    t1 = utime.ticks_cpu()
    for i in range(100):
        tsss = utime.ticks_cpu()
        #reading = analog_value.read_u16()#*conversion_factor
        x[i] = analog_value.read_u16()
    t1 = utime.ticks_cpu()
    print(t1-t0)
    t0 = t1
    print(max(x))



# import rp_devices as devs
# import array
# import uctypes, time

# ADC_CHAN = 0
# ADC_PIN  = 26 + ADC_CHAN
# adc = devs.ADC_DEVICE
# pin = devs.GPIO_PINS[ADC_PIN]
# pad = devs.PAD_PINS[ADC_PIN]
# pin.GPIO_CTRL_REG = devs.GPIO_FUNC_NULL
# pad.PAD_REG = 0

# adc.CS_REG = adc.FCS_REG = 0
# adc.CS.EN = 1
# adc.CS.AINSEL = ADC_CHAN

# adc.CS.START_ONCE = 1
# print(adc.RESULT_REG)


# DMA_CHAN = 0
# NSAMPLES = 100
# RATE = 16000
# dma_chan = devs.DMA_CHANS[DMA_CHAN]
# dma = devs.DMA_DEVICE

# adc.FCS.EN = adc.FCS.DREQ_EN = 1
# adc_buff = array.array('H', (0 for _ in range(NSAMPLES)))
# adc.DIV_REG = (48000000 // RATE - 1) << 8
# adc.FCS.THRESH = adc.FCS.OVER = adc.FCS.UNDER = 1

# dma_chan.READ_ADDR_REG = devs.ADC_FIFO_ADDR
# dma_chan.WRITE_ADDR_REG = uctypes.addressof(adc_buff)
# dma_chan.TRANS_COUNT_REG = NSAMPLES
# dma_chan.CTRL_TRIG_REG = 0
# dma_chan.CTRL_TRIG.CHAIN_TO = DMA_CHAN
# dma_chan.CTRL_TRIG.INCR_WRITE = dma_chan.CTRL_TRIG.IRQ_QUIET = 1
# dma_chan.CTRL_TRIG.TREQ_SEL = devs.DREQ_ADC
# dma_chan.CTRL_TRIG.DATA_SIZE = 1
# dma_chan.CTRL_TRIG.EN = 1



# while adc.FCS.LEVEL:
#     x = adc.FIFO_REG


# adc.CS.START_MANY = 1
# while dma_chan.CTRL_TRIG.BUSY:
#     time.sleep_ms(10)
# adc.CS.START_MANY = 0
# dma_chan.CTRL_TRIG.EN = 0
# vals = [("%1.3f" % (val*3.3/4096)) for val in adc_buff]
# print(vals)


