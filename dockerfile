FROM alpine:latest

RUN apk --no-cache add openssl git curl openssh-client bash


COPY . /tmp
WORKDIR /tmp

RUN ls
RUN chmod +x /tmp/command.sh
ENTRYPOINT [ "/command.sh" ]

RUN echo Current Directory: `pwd`\
    && mkdir temp\
    && cd temp\
    && curl -sLO https://github.com/git-lfs/git-lfs/releases/download/v2.6.0/git-lfs-linux-amd64-v2.6.0.tar.gz \
    && tar -zxf git-lfs-linux-amd64-v2.6.0.tar.gz \
    && ./install.sh \
    && cd / \
    && rm -rf temp