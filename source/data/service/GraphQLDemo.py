# coding=gbk
from graphene import ObjectType, Schema
import graphene


# class Query(ObjectType):
#     hello = graphene.String(name=graphene.String(default_value="aaa", required=True))
#
#     @staticmethod
#     def resolve_hello(root, info, name):
#         return f"hellw word -- {name}"


# schema = Schema(query=Query)
from source.data.service.GraphqlHelper import GraphqlHelper

if __name__ == '__main__':
    # query_string = '''{hello(name:"aaa")}'''
    # result = schema.execute(query_string)
    # print(result.data['hello'])
    print(GraphqlHelper.STR_KEY_QUERY_PR_TIMELINE)
