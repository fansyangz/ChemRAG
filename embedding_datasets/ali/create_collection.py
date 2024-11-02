import dashvector

client = dashvector.Client(
    api_key='sk-ZcecWF1ixQZb91naBt8YuZNmQiV22FA1ADDF3427B11EF93DD762CA87E1E4C',
    endpoint='vrs-cn-g6z3tub070005e.dashvector.cn-hangzhou.aliyuncs.com'
)
collection = client.get(name="vector")

if __name__ == '__main__':
    # 创建一个名称为quickstart、向量维度为4、
    # 向量数据类型为float（默认）、
    # 距离度量方式为dotproduct（内积）的Collection
    # 并预先定义三个Field，名称为name、weight、age，数据类型分别为str、float、int
    # timeout为-1 ,开启create接口异步模式
    ret = client.create(
        name='vector',
        dimension=4096,
        dtype=float,
        fields_schema={'origin_text': str, 'cmpdname': str},
        timeout=-1
    )


    # 判断collection是否创建成功
    if ret:
        print('create collection success!')
    # 等同于下列代码
    # from dashvector import DashVectorCode
    # if ret.code == DashVectorCode.Success:
    #     print('create collection success!')