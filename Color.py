class Color:
    @staticmethod
    def hexToRGB(hexSTR):
        return [int(hexSTR[i:i + 2], 16) for i in range(1, 6, 2)]

    @staticmethod
    def getColorGradient(c1, c2, n):
        import numpy as np
        assert n > 1
        c1_rgb = np.array(Color.hexToRGB(c1)) / 255
        c2_rgb = np.array(Color.hexToRGB(c2)) / 255
        mix_pcts = [x / (n - 1) for x in range(n)]
        rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
        return ["#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors]
