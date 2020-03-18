# coding=gbk


class StringKeyUtils:
    """该类用于存放所有使用的数据项key"""

    '''项目信息使用的key
    '''
    STR_KEY_ID = 'id'
    STR_KEY_NUMBER = 'number'
    STR_KEY_LANG = 'language'
    STR_KEY_OWNER = 'owner'
    STR_KEY_LANG_OTHER = 'Other'
    STR_KEY_NODE_ID = 'node_id'
    STR_KEY_NAME = 'name'
    STR_KEY_FULL_NAME = 'full_name'
    STR_KEY_DESCRIPTION = 'description'
    STR_KEY_CREATE_AT = 'created_at'
    STR_KEY_UPDATE_AT = 'updated_at'
    STR_KEY_STARGAZERS_COUNT = 'stargazers_count'
    STR_KEY_WATCHERS_COUNT = 'watchers_count'
    STR_KEY_LANGUAGE = 'language'
    STR_KEY_FORKS_COUNT = 'forks_count'
    STR_KEY_SUBSCRIBERS_COUNT = 'subscribers_count'
    STR_KEY_OWNER_LOGIN = 'owner_login'
    STR_KEY_PARENT_FULL_NAME = 'parent_full_name'
    STR_KEY_PARENT = 'parent'

    '''用户信息使用到的key '''
    STR_KEY_LOGIN = 'login'
    STR_KEY_SITE_ADMIN = 'site_admin'
    STR_KEY_TYPE = 'type'
    STR_KEY_EMAIL = 'email'
    STR_KEY_FOLLOWERS_URL = 'followers_url'
    STR_KEY_FOLLOWING_URL = 'following_url'
    STR_KEY_STARRED_URL = 'starred_url'
    STR_KEY_SUBSCRIPTIONS_URL = 'subscriptions_url'
    STR_KEY_ORGANIZATIONS_URL = 'organizations_url'
    STR_KEY_REPOS_URL = 'repos_url'
    STR_KEY_EVENTS_URL = 'events_url'
    STR_KEY_RECEVIED_EVENTS_URL = 'received_events_url'
    STR_KEY_COMPANY = 'company'
    STR_KEY_BLOG = 'blog'
    STR_KEY_LOCATION = 'location'
    STR_KEY_HIREABLE = 'hireable'
    STR_KEY_BIO = 'bio'
    STR_KEY_PUBLIC_REPOS = 'public_repos'
    STR_KEY_PUBLIC_GISTS = 'public_gists'
    STR_KEY_FOLLOWERS = 'followers'
    STR_KEY_FOLLOWING = 'following'

    '''pull request可能会使用到的信息'''
    STR_KEY_STATE = 'state'
    STR_KEY_TITLE = 'title'
    STR_KEY_USER = 'user'
    STR_KEY_BODY = 'body'
    STR_KEY_CLOSED_AT = 'closed_at'
    STR_KEY_MERGED_AT = 'merged_at'
    STR_KEY_MERGE_COMMIT_SHA = 'merge_commit_sha'
    STR_KEY_AUTHOR_ASSOCIATION = 'author_association'
    STR_KEY_MERGED = 'merged'
    STR_KEY_COMMENTS = 'comments'
    STR_KEY_REVIEW_COMMENTS = 'review_comments'
    STR_KEY_COMMITS = 'commits'
    STR_KEY_ADDITIONS = 'additions'
    STR_KEY_DELETIONS = 'deletions'
    STR_KEY_CHANGED_FILES = 'changed_files'
    STR_KEY_HEAD = 'head'
    STR_KEY_BASE = 'base'
    STR_KEY_USER_ID = 'user_id'
    STR_KEY_BASE_LABEL = 'base_label'
    STR_KEY_HEAD_LABEL = 'head_label'
    STR_KEY_REPO_FULL_NAME = 'repo_full_name'

    '''Branch 可能会使用的数据'''
    STR_KEY_LABEL = 'label'
    STR_KEY_REF = 'ref'
    STR_KEY_REPO = 'repo'
    STR_KEY_SHA = 'sha'
    STR_KEY_USER_LOGIN = 'user_login'

    '''review可能会使用放日数据'''
    STR_KEY_PULL_NUMBER = 'pull_number'
    STR_KEY_SUBMITTED_AT = 'submitted_at'
    STR_KEY_COMMIT_ID = 'commit_id'

    '''review comment 可能会用到的数据'''
    STR_KEY_PULL_REQUEST_REVIEW_ID = 'pull_request_review_id'
    STR_KEY_DIFF_HUNK = 'diff_hunk'
    STR_KEY_PATH = 'path'
    STR_KEY_POSITION = 'position'
    STR_KEY_ORIGINAL_POSITION = 'original_position'
    STR_KEY_ORIGINAL_COMMIT_ID = 'original_commit_id'
    STR_KEY_START_LINE = 'start_line'
    STR_KEY_ORIGINAL_START_LINE = 'original_start_line'
    STR_KEY_START_SIDE = 'start_side'
    STR_KEY_LINE = 'line'
    STR_KEY_ORIGINAL_LINE = 'original_line'
    STR_KEY_SIDE = 'side'
    STR_KEY_IN_REPLY_TO_ID = 'in_reply_to_id'

    '''issue comment 可能会使用的数据'''

    '''commit 可能会使用的数据'''
    STR_KEY_COMMIT = 'commit'
    STR_KEY_AUTHOR = 'author'
    STR_KEY_DATE = 'date'
    STR_KEY_AUTHOR_LOGIN = 'author_login'
    STR_KEY_COMMITTER = 'committer'
    STR_KEY_COMMITTER_LOGIN = 'committer_login'
    STR_KEY_COMMIT_AUTHOR_DATE = 'commit_author_date'
    STR_KEY_COMMIT_COMMITTER_DATE = 'commit_committer_date'
    STR_KEY_MESSAGE = 'message'
    STR_KEY_COMMIT_MESSAGE = 'commit_message'
    STR_KEY_COMMENT_COUNT = 'comment_count'
    STR_KEY_COMMIT_COMMENT_COUNT = 'commit_comment_count'
    STR_KEY_STATS = 'stats'
    STR_KEY_STATUS = 'status'  # 一个使用在commit一个使用在file
    STR_KEY_TOTAL = 'total'
    STR_KEY_STATUS_TOTAL = 'status_total'
    STR_KEY_STATUS_ADDITIONS = 'status_additions'
    STR_KEY_STATUS_DELETIONS = 'status_deletions'
    STR_KEY_PARENTS = 'parents'
    STR_KEY_FILES = 'files'

    '''file 可能会使用的数据'''
    STR_KEY_COMMIT_SHA = 'commit_sha'
    STR_KEY_CHANGES = 'changes'
    STR_KEY_FILENAME = 'filename'
    STR_KEY_PATCH = 'patch'

    '''commit relation 可能使用的数据'''
    STR_KEY_CHILD = 'child'

    '''设置代理可能会使用到的key'''
    STR_PROXY_HTTP = 'http'
    STR_PROXY_HTTP_FORMAT = 'http://{}'

    '''column做屏蔽可能会使用到的key'''
    STR_KEY_NOP = ''

    '''pr timelineItem 可能会使用到的'''
    STR_KEY_PULL_REQUEST_NODE = 'pullrequest_node'
    STR_KEY_TIME_LINE_ITEM_NODE = 'timelineitem_node'
    STR_KEY_TYPE_NAME_JSON = '__typename'
    STR_KEY_EDGE = 'edge'
    STR_KEY_TYPE_NAME = 'typename'
    STR_KEY_DATA = 'data'
    STR_KEY_NODES = 'nodes'
    STR_KEY_NODE = 'node'
    STR_KEY_TIME_LINE_ITEMS = 'timelineItems'
    STR_KEY_EDGES = 'edges'
    STR_KEY_OID = 'oid'
    STR_FAILED_FETCH = 'Failed to fetch'



    '''HeadRefForcePushedEvent 可能会使用到的'''
    STR_KEY_AFTER_COMMIT = 'afterCommit'
    STR_KEY_BEFORE_COMMIT = 'beforeCommit'
    STR_KEY_HEAD_REF_PUSHED_EVENT = 'HeadRefForcePushedEvent'

    '''PullRequestCommit 可能会使用到的'''
    STR_KEY_PULL_REQUEST_COMMIT = 'PullRequestCommit'

    '''time line item 可能会碰到的其他类型'''
    STR_KEY_ISSUE_COMMIT = 'IssueComment'
    STR_KEY_MENTIONED_EVENT = 'MentionedEvent'
    STR_KEY_PULL_REQUEST_REVIEW = 'PullRequestReview'



    API_GITHUB = 'https://api.github.com'
    API_REVIEWS_FOR_PULL_REQUEST = '/repos/:owner/:repo/pulls/:pull_number/reviews'
    API_PULL_REQUEST_FOR_PROJECT = '/repos/:owner/:repo/pulls'
    API_COMMENTS_FOR_REVIEW = '/repos/:owner/:repo/pulls/:pull_number/reviews/:review_id/comments'
    API_COMMENTS_FOR_PULL_REQUEST = '/repos/:owner/:repo/pulls/:pull_number/comments'
    API_PULL_REQUEST = '/repos/:owner/:repo/pulls/:pull_number'
    API_PROJECT = '/repos/:owner/:repo'
    API_USER = '/users/:user'
    API_REVIEW = '/repos/:owner/:repo/pulls/:pull_number/reviews/:review_id'
    API_ISSUE_COMMENT_FOR_ISSUE = '/repos/:owner/:repo/issues/:issue_number/comments'
    API_COMMIT = '/repos/:owner/:repo/commits/:commit_sha'
    API_COMMITS_FOR_PULL_REQUEST = '/repos/:owner/:repo/pulls/:pull_number/commits'
    API_COMMIT_COMMENTS_FOR_COMMIT = '/repos/:owner/:repo/commits/:commit_sha/comments'
    API_GRAPHQL = '/graphql'

    # 用于替换的字符串
    STR_HEADER_AUTHORIZAITON = 'Authorization'
    STR_HEADER_TOKEN = 'token '  # 有空格
    STR_HEADER_ACCEPT = 'Accept'
    STR_HEADER_MEDIA_TYPE = 'application/vnd.github.comfort-fade-preview+json'
    STR_HEADER_RATE_LIMIT_REMIAN = 'X-RateLimit-Remaining'
    STR_HEADER_RATE_LIMIT_RESET = 'X-RateLimit-Reset'
    STR_HEADER_USER_AGENT = 'User-Agent'
    STR_HEADER_USER_AGENT_SET = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
                                '(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
    USER_AGENTS = [
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
        "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
        "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
        "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
        "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5"
    ]

    STR_OWNER = ':owner'
    STR_REPO = ':repo'
    STR_PULL_NUMBER = ':pull_number'
    STR_REVIEW_ID = ':review_id'
    STR_USER = ':user'
    STR_ISSUE_NUMBER = ':issue_number'
    STR_COMMIT_SHA = ':commit_sha'

    STR_PARM_STARE = 'state'
    STR_PARM_ALL = 'all'
    STR_PARM_OPEN = 'open'
    STR_PARM_CLOSED = 'closed'

    RATE_LIMIT = 5

    """json 404处理用到的"""
    STR_NOT_FIND = 'Not Found'

    """日期转换用到的"""
    STR_STYLE_DATA_DATE = '%Y-%m-%dT%H:%M:%SZ'

    """tsv 文件使用到的"""
    STR_SPLIT_SEP_TSV = '\t'

    """做路径分割可能需要的"""
    STR_SPLIT_SEP_ONE = '\\'
    STR_SPLIT_SEP_TWO = '/'

    """graphql 可能用到的"""
    STR_KEY_QUERY = 'query'
    STR_KEY_OPERATIONAME = 'operationName'
    STR_KEY_VARIABLES = 'variables'


